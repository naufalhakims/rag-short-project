(() => {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const ui = {
        sidebar: $('sidebar'), backdrop: $('backdrop'),
        sessions: $('session-list'), documents: $('document-list'), scopeNote: $('scope-note'),
        dropZone: $('drop-zone'), fileInput: $('file-input'), uploadHint: $('upload-hint'),
        uploadStatus: $('upload-status'),
        title: $('chat-title'), pill: $('model-pill'), modelLabel: $('model-label'), notice: $('notice'),
        messages: $('messages'), thread: $('thread'),
        composer: $('composer'), question: $('question'), send: $('send'), sendIcon: $('send-icon'),
    };

    const state = {
        sessionId: null,
        streaming: false,
        abort: null,
        documents: [],
        excluded: new Set(),   // ids of documents left out of the search
        healthTimer: null,
    };

    const STAGES = {
        rewriting: 'Understanding the question…',
        searching: 'Searching your library…',
        generating: 'Writing the answer…',
    };

    // ── Small helpers ────────────────────────────────────────────

    /** Create an element: el('div', {class: 'x', onclick: fn}, child, 'text'…) */
    function el(tag, attrs = {}, ...children) {
        const node = document.createElement(tag);
        for (const [key, value] of Object.entries(attrs)) {
            if (value == null || value === false) continue;
            if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
            else if (key === 'class') node.className = value;
            else node.setAttribute(key, value === true ? '' : value);
        }
        node.append(...children.filter((c) => c != null));
        return node;
    }

    function icon(name) {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'icon');
        const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
        use.setAttribute('href', `#i-${name}`);
        svg.append(use);
        return svg;
    }

    async function api(path, options) {
        const response = await fetch(path, options);
        if (!response.ok) throw new Error(await errorDetail(response));
        return response.json();
    }

    async function errorDetail(response) {
        try {
            const body = await response.json();
            const detail = body.detail;
            if (typeof detail === 'string') return detail;
            if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;
        } catch { /* not JSON */ }
        return `Request failed (${response.status}).`;
    }

    function relativeDay(iso) {
        const date = new Date(iso);
        if (Number.isNaN(date.getTime())) return '';
        const days = Math.round((new Date().setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) / 864e5);
        if (days === 0) return 'Today';
        if (days === 1) return 'Yesterday';
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }

    const shortModel = (model) => (model || '').split('/').pop();

    function nearBottom() {
        const box = ui.messages;
        return box.scrollHeight - box.scrollTop - box.clientHeight < 140;
    }

    function scrollToBottom(force = false) {
        if (force || nearBottom()) ui.messages.scrollTop = ui.messages.scrollHeight;
    }

    // ── Markdown and citations ───────────────────────────────────

    const CITATION = /\[(\d+(?:\s*[,;]\s*\d+)*)\]/g;

    /** Render Markdown to a sanitised fragment; "[2]" becomes a button when source 2 exists. */
    function renderMarkdown(text, sourceCount = 0) {
        const template = document.createElement('template');
        if (window.marked && window.DOMPurify) {
            template.innerHTML = DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: true }));
        } else {
            template.content.append(el('p', { style: 'white-space: pre-wrap' }, text));
        }
        const root = template.content;
        root.querySelectorAll('a').forEach((a) => {
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
        });
        if (sourceCount > 0) linkCitations(root, sourceCount);
        return root;
    }

    function linkCitations(root, sourceCount) {
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        const targets = [];
        while (walker.nextNode()) {
            const node = walker.currentNode;
            if (!node.parentElement?.closest('code, pre, a') && node.nodeValue.includes('[')) targets.push(node);
        }
        for (const node of targets) {
            const parts = [];
            let last = 0;
            for (const match of node.nodeValue.matchAll(CITATION)) {
                const numbers = match[1].split(/[,;]/).map((n) => parseInt(n, 10));
                if (!numbers.every((n) => n >= 1 && n <= sourceCount)) continue;
                parts.push(node.nodeValue.slice(last, match.index));
                for (const n of numbers) {
                    parts.push(el('button', {
                        class: 'cite', type: 'button', 'data-cite': n, title: `Show source ${n}`,
                    }, String(n)));
                }
                last = match.index + match[0].length;
            }
            if (!parts.length) continue;
            parts.push(node.nodeValue.slice(last));
            node.replaceWith(...parts.filter((p) => p !== ''));
        }
    }

    // ── Messages ─────────────────────────────────────────────────

    function addUserMessage(text) {
        const node = el('div', { class: 'msg user' }, el('div', { class: 'bubble' }, text));
        ui.thread.append(node);
        scrollToBottom(true);
        return node;
    }

    function sourceLabel(source) {
        const section = (source.heading || '').split(' > ').pop();
        return [source.filename, section, source.page ? `p. ${source.page}` : ''].filter(Boolean).join(' · ');
    }

    /** An assistant message that can be filled in while streaming or all at once. */
    function createAssistantMessage() {
        const body = el('div', { class: 'prose' });
        const status = el('div', { class: 'status' }, el('span', { class: 'spinner' }), el('span', {}, 'Thinking…'));
        const footer = el('div', { class: 'msg-footer' });
        const main = el('div', { class: 'msg-main' }, status, body, footer);
        const root = el('div', { class: 'msg assistant' }, el('div', { class: 'avatar' }, icon('book')), main);
        ui.thread.append(root);

        let text = '';
        let sources = [];
        let frame = null;
        let showAll = false;
        let openSource = null;

        function paint(streaming) {
            frame = null;
            body.replaceChildren(renderMarkdown(text, sources.length));
            body.classList.toggle('caret', streaming);
            scrollToBottom();
        }

        function renderFooter(meta = {}) {
            const cited = sources.filter((s) => s.cited);
            const visible = showAll ? sources : cited;
            const chips = visible.map((source) => el('button', {
                class: `chip${source.cited ? '' : ' uncited'}${openSource === source.n ? ' open' : ''}`,
                type: 'button',
                title: source.text ? 'Show this passage' : sourceLabel(source),
                onclick: () => toggleSource(source.n),
            }, el('span', { class: 'num' }, String(source.n)), el('span', { class: 'label' }, sourceLabel(source))));

            const shown = sources.find((s) => s.n === openSource && visible.includes(s));
            const passage = shown && el('div', { class: 'passage' },
                el('div', { class: 'passage-head' },
                    el('strong', {}, shown.filename),
                    shown.heading ? el('span', {}, shown.heading) : null,
                    shown.page ? el('span', {}, `Page ${shown.page}`) : null),
                el('p', { class: 'passage-text' }, shown.text || 'The passage text was not saved for this older answer.'));

            const details = [];
            if (meta.model) details.push(el('span', { title: `${meta.provider} · ${meta.model}` }, shortModel(meta.model)));
            if (meta.timings_ms?.total) details.push(el('span', {}, `${(meta.timings_ms.total / 1000).toFixed(1)} s`));
            if (meta.stopped) details.push(el('span', {}, 'Stopped'));
            if (sources.length > cited.length) {
                details.push(el('button', {
                    class: 'link-btn', type: 'button',
                    onclick: () => { showAll = !showAll; renderFooter(meta); },
                }, showAll ? 'Show cited sources only' : `Show all ${sources.length} passages searched`));
            }
            if (text) {
                const copy = el('button', { class: 'link-btn', type: 'button' }, icon('copy'), 'Copy');
                copy.addEventListener('click', async () => {
                    try { await navigator.clipboard.writeText(text); } catch { return; }
                    copy.replaceChildren(icon('check'), 'Copied');
                    setTimeout(() => copy.replaceChildren(icon('copy'), 'Copy'), 1500);
                });
                details.push(copy);
            }

            // replaceChildren() would render a null argument as the text "null".
            footer.replaceChildren(...[
                chips.length ? el('div', { class: 'source-chips' }, el('span', { class: 'source-label' }, 'Sources'), ...chips) : null,
                passage,
                details.length ? el('div', { class: 'msg-meta' }, ...details) : null,
            ].filter(Boolean));
            footer._meta = meta;
        }

        function toggleSource(n, forceOpen = false) {
            const source = sources.find((s) => s.n === n);
            if (!source) return;
            if (!source.cited) showAll = true;
            openSource = openSource === n && !forceOpen ? null : n;
            renderFooter(footer._meta);
            if (openSource) footer.querySelector('.passage')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }

        body.addEventListener('click', (event) => {
            const cite = event.target.closest('[data-cite]');
            if (cite) toggleSource(Number(cite.dataset.cite), true);
        });

        return {
            root,
            get text() { return text; },
            setStatus(stage) {
                if (!text) status.lastChild.textContent = STAGES[stage] || 'Working…';
            },
            setSources(list) { sources = list || []; },
            append(delta) {
                text += delta;
                status.hidden = true;
                if (!frame) frame = requestAnimationFrame(() => paint(true));
            },
            finish({ cited = null, meta = {} } = {}) {
                if (frame) cancelAnimationFrame(frame);
                status.hidden = true;
                if (cited) sources.forEach((s) => { s.cited = cited.includes(s.n); });
                paint(false);
                renderFooter(meta);
            },
            fail(message, onRetry) {
                if (frame) cancelAnimationFrame(frame);
                status.hidden = true;
                body.classList.remove('caret');
                body.replaceChildren();
                text = '';
                footer.replaceChildren(el('div', { class: 'error-box', role: 'alert' },
                    el('p', {}, message),
                    onRetry ? el('button', { class: 'btn btn-ghost', type: 'button', onclick: onRetry }, icon('retry'), 'Try again') : null));
                scrollToBottom();
            },
        };
    }

    function showWelcome() {
        const hasDocs = state.documents.length > 0;
        const suggestions = [
            ['Summarize', 'Get the main ideas of your documents.', 'Summarize the main ideas in my documents.'],
            ['Key facts', 'Pull out names, numbers and dates.', 'What are the most important facts, figures and dates in my documents?'],
            state.documents.length > 1
                ? ['Compare', 'See where documents agree or differ.', 'Compare my documents: where do they overlap and where do they differ?']
                : ['Quiz me', 'Test what you remember.', 'Write five quiz questions, with answers, based on my document.'],
        ];
        ui.thread.replaceChildren(el('div', { class: 'welcome' },
            el('div', { class: 'welcome-mark' }, icon('book')),
            el('h2', {}, 'Ask your library anything.'),
            el('p', {}, hasDocs
                ? 'Answers come from your uploaded documents, with numbered citations you can open to check the source.'
                : 'Add a PDF, Word, Markdown or text file to get started. Answers will come from your documents, with citations.'),
            hasDocs
                ? el('div', { class: 'suggestions' }, ...suggestions.map(([title, hint, prompt]) =>
                    el('button', { class: 'suggestion', type: 'button', onclick: () => send(prompt) },
                        el('strong', {}, title), el('span', {}, hint))))
                : el('button', { class: 'btn btn-primary', type: 'button', onclick: () => ui.fileInput.click() },
                    icon('upload'), 'Add documents'),
        ));
    }

    const isWelcomeShown = () => Boolean(ui.thread.querySelector('.welcome'));

    // ── Chat ─────────────────────────────────────────────────────

    /** Yield the JSON events of a Server-Sent Events response. */
    async function* readEvents(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
            const { done, value } = await reader.read();
            if (done) return;
            buffer += decoder.decode(value, { stream: true });
            let end;
            while ((end = buffer.indexOf('\n\n')) !== -1) {
                const block = buffer.slice(0, end);
                buffer = buffer.slice(end + 2);
                for (const line of block.split('\n')) {
                    if (!line.startsWith('data: ')) continue;
                    try { yield JSON.parse(line.slice(6)); } catch { /* skip malformed event */ }
                }
            }
        }
    }

    function searchScope() {
        if (!state.excluded.size) return null;
        return state.documents.filter((d) => !state.excluded.has(d.id)).map((d) => d.id);
    }

    async function send(question) {
        question = question.trim();
        if (!question || state.streaming) return;
        if (isWelcomeShown()) ui.thread.replaceChildren();
        closeSidebar();

        const userNode = addUserMessage(question);
        const message = createAssistantMessage();
        const retry = () => { userNode.remove(); message.root.remove(); send(question); };
        scrollToBottom(true);

        state.streaming = true;
        state.abort = new AbortController();
        updateSendButton();
        let settled = false;

        try {
            const body = { question, session_id: state.sessionId };
            const scope = searchScope();
            if (scope) body.doc_ids = scope;
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: state.abort.signal,
            });
            if (!response.ok) {
                message.fail(await errorDetail(response), retry);
                settled = true;
            } else {
                for await (const event of readEvents(response)) {
                    if (event.type === 'session') {
                        state.sessionId = event.session_id;
                        ui.title.textContent = event.title;
                    } else if (event.type === 'status') message.setStatus(event.stage);
                    else if (event.type === 'sources') message.setSources(event.sources);
                    else if (event.type === 'token') message.append(event.text);
                    else if (event.type === 'done') { message.finish(event); settled = true; }
                    else if (event.type === 'error') { message.fail(event.message, retry); settled = true; }
                }
            }
            if (!settled) message.fail('The connection closed before the answer finished.', retry);
        } catch (error) {
            if (error.name === 'AbortError') {
                // Stopped by the user. Keep what was written; the server saves it too.
                if (message.text) message.finish({ meta: { stopped: true } });
                else { userNode.remove(); message.root.remove(); if (!ui.thread.children.length) showWelcome(); }
            } else {
                message.fail('Could not reach the server. Is it still running?', retry);
            }
        } finally {
            state.streaming = false;
            state.abort = null;
            updateSendButton();
            setTimeout(refreshSessions, 250);   // let a stopped answer finish saving first
            ui.question.focus();
        }
    }

    function updateSendButton() {
        const stop = state.streaming;
        ui.send.classList.toggle('stop', stop);
        ui.send.setAttribute('aria-label', stop ? 'Stop generating' : 'Send');
        ui.sendIcon.setAttribute('href', stop ? '#i-stop' : '#i-send');
        ui.send.disabled = !stop && !ui.question.value.trim();
    }

    function autoGrow() {
        ui.question.style.height = 'auto';
        ui.question.style.height = `${Math.min(ui.question.scrollHeight, 200)}px`;
    }

    function startNewChat() {
        if (state.streaming) state.abort?.abort();
        state.sessionId = null;
        ui.title.textContent = 'New chat';
        showWelcome();
        renderSessions(state.sessions || []);
        closeSidebar();
        ui.question.focus();
    }

    // ── Conversations ────────────────────────────────────────────

    async function refreshSessions() {
        try {
            state.sessions = await api('/api/history');
            renderSessions(state.sessions);
        } catch (error) {
            console.error('Could not load conversations:', error);
        }
    }

    function renderSessions(sessions) {
        if (!sessions.length) {
            ui.sessions.replaceChildren(el('li', { class: 'list-empty' }, 'No conversations yet.'));
            return;
        }
        ui.sessions.replaceChildren(...sessions.map((session) => el('li', {
            class: `row${session.id === state.sessionId ? ' active' : ''}`,
        },
            el('button', { class: 'row-open', type: 'button', onclick: () => openSession(session) },
                icon('chat'),
                el('span', { class: 'row-body' },
                    el('span', { class: 'row-title' }, session.title || 'Untitled chat'),
                    el('span', { class: 'row-meta' }, relativeDay(session.updated_at)))),
            el('button', {
                class: 'icon-btn row-delete', type: 'button', 'aria-label': `Delete conversation: ${session.title}`,
                onclick: () => deleteSession(session),
            }, icon('trash')),
        )));
    }

    async function openSession(session) {
        if (state.streaming) state.abort?.abort();
        try {
            const messages = await api(`/api/history/${encodeURIComponent(session.id)}`);
            state.sessionId = session.id;
            ui.title.textContent = session.title || 'Chat';
            ui.thread.replaceChildren();
            for (const item of messages) {
                if (item.role === 'user') { addUserMessage(item.content); continue; }
                const message = createAssistantMessage();
                message.setSources(item.sources || []);
                message.append(item.content);
                message.finish({ meta: item.meta || {} });
            }
            renderSessions(state.sessions);
            closeSidebar();
            scrollToBottom(true);
        } catch (error) {
            alert(`Could not open this conversation: ${error.message}`);
            refreshSessions();
        }
    }

    async function deleteSession(session) {
        if (!confirm(`Delete the conversation "${session.title}"?`)) return;
        try {
            await api(`/api/history/${encodeURIComponent(session.id)}`, { method: 'DELETE' });
        } catch (error) {
            alert(`Could not delete: ${error.message}`);
        }
        if (session.id === state.sessionId) startNewChat();
        refreshSessions();
    }

    // ── Documents ────────────────────────────────────────────────

    async function refreshDocuments() {
        try {
            state.documents = await api('/api/documents');
        } catch (error) {
            console.error('Could not load documents:', error);
            return;
        }
        const ids = new Set(state.documents.map((d) => d.id));
        state.excluded.forEach((id) => { if (!ids.has(id)) state.excluded.delete(id); });
        renderDocuments();
        if (isWelcomeShown()) showWelcome();
    }

    function renderDocuments() {
        const docs = state.documents;
        const included = docs.length - state.excluded.size;
        ui.scopeNote.hidden = !state.excluded.size;
        ui.scopeNote.textContent = `Searching ${included} of ${docs.length}`;
        if (!docs.length) {
            ui.documents.replaceChildren(el('li', { class: 'list-empty' }, 'No documents yet.'));
            return;
        }
        ui.documents.replaceChildren(...docs.map((doc) => {
            const checkbox = el('input', {
                type: 'checkbox', title: 'Include in search', 'aria-label': `Search in ${doc.filename}`,
            });
            checkbox.checked = !state.excluded.has(doc.id);
            checkbox.addEventListener('change', () => {
                if (checkbox.checked) state.excluded.delete(doc.id);
                else if (included > 1) state.excluded.add(doc.id);   // keep at least one document searchable
                renderDocuments();
            });
            return el('li', { class: `row${state.excluded.has(doc.id) ? ' excluded' : ''}`, title: doc.filename },
                checkbox,
                el('span', { class: 'row-body' },
                    el('span', { class: 'row-title' }, doc.filename),
                    el('span', { class: 'row-meta' }, `${doc.chunk_count} ${doc.chunk_count === 1 ? 'passage' : 'passages'}`)),
                el('button', {
                    class: 'icon-btn row-delete', type: 'button', 'aria-label': `Delete ${doc.filename}`,
                    onclick: () => deleteDocument(doc),
                }, icon('trash')));
        }));
    }

    async function deleteDocument(doc) {
        if (!confirm(`Remove "${doc.filename}" from your library?`)) return;
        try {
            await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: 'DELETE' });
        } catch (error) {
            alert(`Could not delete: ${error.message}`);
        }
        refreshDocuments();
    }

    async function uploadFiles(files) {
        for (const file of files) {
            const state_ = el('span', { class: 'state' });
            const name = el('span', { class: 'name' }, file.name);
            const item = el('li', { class: 'upload-item' }, el('span', { class: 'spinner' }), name, state_);
            ui.uploadStatus.append(item);
            const form = new FormData();
            form.append('file', file);
            try {
                const result = await api('/api/documents', { method: 'POST', body: form });
                const count = result.document.chunk_count;
                item.className = 'upload-item ok';
                item.firstChild.replaceWith(icon('check'));
                state_.textContent = { added: `${count} ${count === 1 ? 'passage' : 'passages'}`, replaced: 'Updated', unchanged: 'Already added' }[result.status];
                setTimeout(() => item.remove(), 5000);
            } catch (error) {
                item.className = 'upload-item error';
                item.firstChild.remove();
                name.textContent = `${file.name}: ${error.message}`;
                item.append(el('button', { class: 'link-btn', type: 'button', 'aria-label': 'Dismiss', onclick: () => item.remove() }, icon('close')));
            }
            await refreshDocuments();
        }
        ui.fileInput.value = '';
    }

    // ── Status ───────────────────────────────────────────────────

    async function refreshHealth() {
        clearTimeout(state.healthTimer);
        let health;
        try {
            health = await api('/api/health');
        } catch {
            setPill('bad', 'Server unreachable', 'The app server is not responding.');
            state.healthTimer = setTimeout(refreshHealth, 5000);
            return;
        }
        const { llm, retrieval, upload } = health;
        ui.fileInput.accept = upload.extensions.join(',');
        ui.uploadHint.textContent = `PDF, Word, Markdown or text · up to ${upload.max_mb} MB`;

        if (!llm.configured || !llm.model) {
            setPill('bad', 'No language model', 'No LLM provider is configured.');
            showNotice('No language model is configured. Add GROQ_API_KEY (free at console.groq.com) to the .env file and restart the app.');
        } else if (retrieval.error) {
            setPill('bad', 'Search models failed', retrieval.error);
            showNotice(`The local search models could not be loaded: ${retrieval.error}`);
        } else if (!retrieval.ready) {
            setPill('busy', 'Loading search models…', 'First start downloads the embedding models (about 150 MB).');
            state.healthTimer = setTimeout(refreshHealth, 2500);
            return;
        } else {
            const others = llm.providers.filter((p) => p.name !== llm.provider && p.model).map((p) => p.name);
            const active = llm.providers.find((p) => p.name === llm.provider);
            const backups = [...(active?.fallback_models || []).map(shortModel), ...others];
            setPill('ok', `${llm.provider} · ${shortModel(llm.model)}`,
                backups.length ? `Falls back to: ${backups.join(', ')}` : 'Language model ready');
            ui.notice.hidden = true;
        }
        state.healthTimer = setTimeout(refreshHealth, 120000);
    }

    function setPill(kind, label, title) {
        ui.pill.className = `pill ${kind}`;
        ui.pill.title = title;
        ui.modelLabel.textContent = label;
    }

    function showNotice(text) {
        ui.notice.textContent = text;
        ui.notice.hidden = false;
    }

    // ── Sidebar (small screens) ──────────────────────────────────

    function openSidebar() { ui.sidebar.classList.add('open'); ui.backdrop.hidden = false; }
    function closeSidebar() { ui.sidebar.classList.remove('open'); ui.backdrop.hidden = true; }

    // ── Wiring ───────────────────────────────────────────────────

    document.addEventListener('click', (event) => {
        const action = event.target.closest('[data-action]')?.dataset.action;
        if (action === 'new-chat') startNewChat();
        else if (action === 'open-sidebar') openSidebar();
        else if (action === 'close-sidebar') closeSidebar();
    });
    ui.backdrop.addEventListener('click', closeSidebar);

    ui.composer.addEventListener('submit', (event) => {
        event.preventDefault();
        if (state.streaming) { state.abort?.abort(); return; }
        const question = ui.question.value;
        if (!question.trim()) return;
        ui.question.value = '';
        autoGrow();
        send(question);
    });
    ui.question.addEventListener('input', () => { autoGrow(); updateSendButton(); });
    ui.question.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            if (!state.streaming) ui.composer.requestSubmit();
        }
    });

    ui.fileInput.addEventListener('change', () => uploadFiles([...ui.fileInput.files]));
    ui.dropZone.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); ui.fileInput.click(); }
    });
    for (const type of ['dragenter', 'dragover']) {
        ui.dropZone.addEventListener(type, (event) => { event.preventDefault(); ui.dropZone.classList.add('dragging'); });
    }
    for (const type of ['dragleave', 'drop']) {
        ui.dropZone.addEventListener(type, (event) => { event.preventDefault(); ui.dropZone.classList.remove('dragging'); });
    }
    ui.dropZone.addEventListener('drop', (event) => uploadFiles([...event.dataTransfer.files]));
    // Dropping a file anywhere else should not navigate away from the app.
    window.addEventListener('dragover', (event) => event.preventDefault());
    window.addEventListener('drop', (event) => event.preventDefault());

    showWelcome();
    updateSendButton();
    refreshDocuments();
    refreshSessions();
    refreshHealth();
    ui.question.focus();
})();
