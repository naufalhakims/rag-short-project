(() => {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const ui = {
        sidebar: $('sidebar'), backdrop: $('backdrop'),
        sessions: $('session-list'), documents: $('document-list'),
        pickerToggle: $('picker-toggle'), picker: $('picker-list'),
        dropZone: $('drop-zone'), fileInput: $('file-input'), uploadHint: $('upload-hint'),
        uploadStatus: $('upload-status'),
        title: $('chat-title'), pill: $('model-pill'), modelLabel: $('model-label'), notice: $('notice'),
        messages: $('messages'), thread: $('thread'),
        composer: $('composer'), question: $('question'), send: $('send'), sendIcon: $('send-icon'),
        mentionMenu: $('mention-menu'), mentionHint: $('mention-hint'),
    };

    const state = {
        sessionId: null,
        streaming: false,
        abort: null,
        library: [],           // every uploaded document
        chatDocs: [],          // the documents this chat searches
        pickerOpen: false,     // "Add from your files" list expanded
        mention: null,         // open @-menu: { start, options, active }
        healthTimer: null,
    };

    const STAGES = {
        rewriting: 'Understanding the question…',
        searching: 'Searching this chat\'s documents…',
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
        const parts = [];
        let last = 0;
        for (const found of findMentions(text)) {
            parts.push(text.slice(last, found.start), el('span', { class: 'mention' }, text.slice(found.start, found.end)));
            last = found.end;
        }
        parts.push(text.slice(last));
        const node = el('div', { class: 'msg user' }, el('div', { class: 'bubble' }, ...parts.filter((p) => p !== '')));
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
        const hasDocs = state.chatDocs.length > 0;
        const reusable = availableToAdd().length > 0;
        const suggestions = [
            ['Summarize', 'Get the main ideas of your documents.', 'Summarize the main ideas in my documents.'],
            ['Key facts', 'Pull out names, numbers and dates.', 'What are the most important facts, figures and dates in my documents?'],
            state.chatDocs.length > 1
                ? ['Compare', 'See where documents agree or differ.', 'Compare my documents: where do they overlap and where do they differ?']
                : ['Quiz me', 'Test what you remember.', 'Write five quiz questions, with answers, based on my document.'],
        ];
        ui.thread.replaceChildren(el('div', { class: 'welcome' },
            el('div', { class: 'welcome-mark' }, icon('book')),
            el('h2', {}, hasDocs ? 'Ask about this chat\'s documents.' : 'Give this chat its documents.'),
            el('p', {}, hasDocs
                ? `Answers come only from the ${state.chatDocs.length === 1 ? 'document' : `${state.chatDocs.length} documents`} in this chat, with numbered citations you can open to check the source.`
                : 'Each chat uses only its own documents, so topics never mix. Upload a PDF, Word, Markdown or text file, or reuse one you uploaded before.'),
            hasDocs
                ? el('div', { class: 'suggestions' }, ...suggestions.map(([title, hint, prompt]) =>
                    el('button', { class: 'suggestion', type: 'button', onclick: () => send(prompt) },
                        el('strong', {}, title), el('span', {}, hint))))
                : el('div', { class: 'welcome-actions' },
                    el('button', { class: 'btn btn-primary', type: 'button', onclick: () => ui.fileInput.click() },
                        icon('upload'), 'Upload to this chat'),
                    reusable ? el('button', { class: 'btn btn-ghost', type: 'button', onclick: () => { setPicker(true); openSidebar(); } },
                        icon('plus'), 'Add from your files') : null),
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
        state.chatDocs = [];
        ui.title.textContent = 'New chat';
        renderDocuments();
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
            const id = encodeURIComponent(session.id);
            const [messages, chatDocs] = await Promise.all([
                api(`/api/history/${id}`), api(`/api/history/${id}/documents`),
            ]);
            state.sessionId = session.id;
            state.chatDocs = chatDocs;
            renderDocuments();
            ui.title.textContent = session.title || 'Chat';
            ui.thread.replaceChildren();
            if (!messages.length) showWelcome();
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

    /** Every chat has its own documents. A chat is created on the server the first time
     *  it needs one (an upload or an added file) so documents can come before questions. */
    async function ensureSession() {
        if (state.sessionId) return state.sessionId;
        const session = await api('/api/history', { method: 'POST' });
        state.sessionId = session.id;
        state.sessions = [session, ...(state.sessions || [])];
        renderSessions(state.sessions);
        return session.id;
    }

    const chatPath = () => `/api/history/${encodeURIComponent(state.sessionId)}/documents`;

    function availableToAdd() {
        const attached = new Set(state.chatDocs.map((d) => d.id));
        return state.library.filter((d) => !attached.has(d.id));
    }

    async function refreshDocuments() {
        try {
            const [library, chatDocs] = await Promise.all([
                api('/api/documents'),
                state.sessionId ? api(chatPath()) : Promise.resolve([]),
            ]);
            state.library = library;
            state.chatDocs = chatDocs;
        } catch (error) {
            console.error('Could not load documents:', error);
            return;
        }
        renderDocuments();
        if (isWelcomeShown()) showWelcome();
    }

    function setPicker(open) {
        state.pickerOpen = open;
        renderDocuments();
    }

    function docMeta(doc) {
        return `${doc.chunk_count} ${doc.chunk_count === 1 ? 'passage' : 'passages'}`;
    }

    function renderDocuments() {
        updateMentionHint();   // tags depend on which documents this chat has
        // This chat's documents: removing one only takes it out of this chat.
        ui.documents.replaceChildren(...(state.chatDocs.length
            ? state.chatDocs.map((doc) => el('li', { class: 'row', title: doc.filename },
                icon('file'),
                el('span', { class: 'row-body' },
                    el('span', { class: 'row-title' }, doc.filename),
                    el('span', { class: 'row-meta' }, docMeta(doc))),
                el('button', {
                    class: 'icon-btn row-delete', type: 'button', title: 'Remove from this chat',
                    'aria-label': `Remove ${doc.filename} from this chat`, onclick: () => detachDocument(doc),
                }, icon('close'))))
            : [el('li', { class: 'list-empty' }, 'None yet. This chat answers only from documents you add here.')]));

        // Files uploaded in other chats, ready to reuse here.
        const available = availableToAdd();
        ui.pickerToggle.hidden = !available.length;
        if (!available.length) state.pickerOpen = false;
        ui.pickerToggle.setAttribute('aria-expanded', String(state.pickerOpen));
        ui.pickerToggle.replaceChildren(icon(state.pickerOpen ? 'close' : 'plus'),
            state.pickerOpen ? 'Close' : `Add from your files (${available.length})`);
        ui.picker.hidden = !state.pickerOpen;
        if (!state.pickerOpen) return;
        ui.picker.replaceChildren(
            el('li', { class: 'picker-note' }, 'Files you uploaded before. Add one to use it in this chat.'),
            ...available.map((doc) => el('li', { class: 'row', title: doc.filename },
                el('button', {
                    class: 'icon-btn row-add', type: 'button', title: 'Add to this chat',
                    'aria-label': `Add ${doc.filename} to this chat`, onclick: () => attachDocument(doc),
                }, icon('plus')),
                el('span', { class: 'row-body' },
                    el('span', { class: 'row-title' }, doc.filename),
                    el('span', { class: 'row-meta' }, docMeta(doc))),
                el('button', {
                    class: 'icon-btn row-delete', type: 'button', title: 'Delete file everywhere',
                    'aria-label': `Delete ${doc.filename} from all chats`, onclick: () => deleteDocument(doc),
                }, icon('trash')))),
        );
    }

    async function attachDocument(doc) {
        try {
            await ensureSession();
            state.chatDocs = await api(chatPath(), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ doc_ids: [doc.id] }),
            });
        } catch (error) {
            alert(`Could not add the file: ${error.message}`);
        }
        renderDocuments();
        if (isWelcomeShown()) showWelcome();
    }

    async function detachDocument(doc) {
        try {
            await api(`${chatPath()}/${encodeURIComponent(doc.id)}`, { method: 'DELETE' });
        } catch (error) {
            alert(`Could not remove the file: ${error.message}`);
        }
        refreshDocuments();
    }

    async function deleteDocument(doc) {
        if (!confirm(`Delete "${doc.filename}" from your files? It will be removed from every chat that uses it.`)) return;
        try {
            await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: 'DELETE' });
        } catch (error) {
            alert(`Could not delete: ${error.message}`);
        }
        refreshDocuments();
    }

    async function uploadFiles(files) {
        if (!files.length) return;
        let sessionId;
        try {
            sessionId = await ensureSession();
        } catch (error) {
            alert(`Could not start a chat for these files: ${error.message}`);
            return;
        }
        for (const file of files) {
            const state_ = el('span', { class: 'state' });
            const name = el('span', { class: 'name' }, file.name);
            const item = el('li', { class: 'upload-item' }, el('span', { class: 'spinner' }), name, state_);
            ui.uploadStatus.append(item);
            const form = new FormData();
            form.append('file', file);
            try {
                const url = `/api/documents?session_id=${encodeURIComponent(sessionId)}`;
                const result = await api(url, { method: 'POST', body: form });
                const count = result.document.chunk_count;
                item.className = 'upload-item ok';
                item.firstChild.replaceWith(icon('check'));
                state_.textContent = { added: `${count} ${count === 1 ? 'passage' : 'passages'}`, replaced: 'Updated', unchanged: 'Added to this chat' }[result.status];
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

    // ── @-mentions ───────────────────────────────────────────────
    //
    // "@file name" in a question limits the search to that document. The server does
    // the real matching (app/rag/mentions.py); this mirrors it for the menu, the hint
    // and highlighting.

    const stem = (filename) => filename.replace(/\.[^.\s]+$/, '');
    const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    /** Mentions of this chat's documents in `text`: [{start, end, doc}], in order. */
    function findMentions(text) {
        const names = [];
        for (const doc of state.chatDocs) {
            names.push([doc.filename, doc]);
            const short = stem(doc.filename);
            if (short && short !== doc.filename) names.push([short, doc]);
        }
        names.sort((a, b) => b[0].length - a[0].length);
        const found = [];
        let masked = text;
        for (const [name, doc] of names) {
            const pattern = new RegExp(`@${escapeRegExp(name)}(?=$|[\\s,;:!?)\\]}"'])`, 'gi');
            masked = masked.replace(pattern, (match, offset) => {
                found.push({ start: offset, end: offset + match.length, doc });
                return ' '.repeat(match.length);
            });
        }
        return found.sort((a, b) => a.start - b.start);
    }

    function updateMentionHint() {
        const docs = [...new Map(findMentions(ui.question.value).map((m) => [m.doc.id, m.doc])).values()];
        ui.mentionHint.hidden = !docs.length;
        if (!docs.length) return;
        ui.mentionHint.replaceChildren('Only searching', ...docs.map((doc) => el('span', { class: 'mention-tag' }, doc.filename)));
    }

    /** The "@query" being typed just before the caret, if any. */
    function mentionQuery() {
        const box = ui.question;
        if (box.selectionStart !== box.selectionEnd) return null;
        const before = box.value.slice(0, box.selectionStart);
        const at = before.lastIndexOf('@');
        if (at === -1 || (at > 0 && !/\s/.test(before[at - 1]))) return null;
        const query = before.slice(at + 1);
        if (query.includes('\n') || query.length > 80) return null;
        return { start: at, query };
    }

    function updateMentionMenu() {
        const typed = mentionQuery();
        if (!typed) return closeMentionMenu();
        const query = typed.query.toLowerCase();
        const options = state.chatDocs.filter((doc) => doc.filename.toLowerCase().includes(query));
        // A space after a query that matches nothing means the user moved on.
        if (!options.length && /\s/.test(typed.query)) return closeMentionMenu();
        const previous = state.mention;
        const active = previous && previous.start === typed.start ? Math.min(previous.active, Math.max(options.length - 1, 0)) : 0;
        state.mention = { start: typed.start, options, active };
        renderMentionMenu();
    }

    function renderMentionMenu() {
        const { options, active } = state.mention;
        ui.mentionMenu.hidden = false;
        if (!state.chatDocs.length) {
            ui.mentionMenu.replaceChildren(el('li', { class: 'mention-empty' }, 'This chat has no documents yet. Upload one or add one from your files.'));
            return;
        }
        if (!options.length) {
            ui.mentionMenu.replaceChildren(el('li', { class: 'mention-empty' }, 'No document in this chat matches.'));
            return;
        }
        ui.mentionMenu.replaceChildren(
            el('li', { class: 'menu-title', role: 'presentation' }, 'Search only in…'),
            ...options.map((doc, index) => {
                const option = el('li', {
                    class: 'mention-option', role: 'option', id: `mention-${index}`,
                    'aria-selected': String(index === active),
                },
                    icon('file'),
                    el('span', { class: 'row-body' },
                        el('span', { class: 'row-title' }, doc.filename),
                        el('span', { class: 'row-meta' }, docMeta(doc))));
                // mousedown, not click: keep focus (and the caret) in the question box.
                option.addEventListener('mousedown', (event) => { event.preventDefault(); chooseMention(doc); });
                return option;
            }));
        ui.question.setAttribute('aria-activedescendant', `mention-${active}`);
        ui.mentionMenu.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' });
    }

    function closeMentionMenu() {
        state.mention = null;
        ui.mentionMenu.hidden = true;
        ui.question.removeAttribute('aria-activedescendant');
    }

    function chooseMention(doc) {
        const box = ui.question;
        const { start } = state.mention;
        const insert = `@${doc.filename} `;
        box.value = box.value.slice(0, start) + insert + box.value.slice(box.selectionStart);
        const caret = start + insert.length;
        box.setSelectionRange(caret, caret);
        closeMentionMenu();
        autoGrow();
        updateSendButton();
        updateMentionHint();
        box.focus();
    }

    /** Keyboard control of the open menu. Returns true when the key was used. */
    function handleMentionKey(event) {
        const menu = state.mention;
        if (!menu) return false;
        if (event.key === 'Escape') { closeMentionMenu(); return true; }
        if (!menu.options.length) return false;
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            const step = event.key === 'ArrowDown' ? 1 : -1;
            menu.active = (menu.active + step + menu.options.length) % menu.options.length;
            renderMentionMenu();
            return true;
        }
        if ((event.key === 'Enter' && !event.shiftKey) || event.key === 'Tab') {
            chooseMention(menu.options[menu.active]);
            return true;
        }
        return false;
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
        closeMentionMenu();
        updateMentionHint();
        send(question);
    });
    ui.question.addEventListener('input', () => { autoGrow(); updateSendButton(); updateMentionMenu(); updateMentionHint(); });
    ui.question.addEventListener('click', updateMentionMenu);
    ui.question.addEventListener('blur', () => setTimeout(closeMentionMenu, 120));
    ui.question.addEventListener('keydown', (event) => {
        if (!event.isComposing && handleMentionKey(event)) { event.preventDefault(); return; }
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            if (!state.streaming) ui.composer.requestSubmit();
        }
    });

    ui.pickerToggle.addEventListener('click', () => setPicker(!state.pickerOpen));
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
