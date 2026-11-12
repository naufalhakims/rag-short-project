# Deploying to Oracle Cloud (Always Free)

This puts Pocket Librarian on a free Oracle Cloud server with a public `https://` address. Your documents and chats are kept on the server's disk and survive restarts and updates.

You will need: a credit or debit card for Oracle's identity check (the Always Free resources are not charged), your Groq API key, and about 45 minutes.

**What runs on the server:** two Docker containers, started by `docker-compose.yml`:

- **app**: Pocket Librarian, built from the `Dockerfile`, with the search models inside the image.
- **caddy**: the public web server. It gets a free HTTPS certificate from Let's Encrypt and passes requests to the app.

---

## 1. Create an Oracle Cloud account

1. Sign up at <https://www.oracle.com/cloud/free/>.
2. Choose your **home region** carefully: it cannot be changed later, and Always Free servers can only be created there. Pick one near you (for Indonesia: Singapore or Batam, if offered).
3. Finish the card verification. Oracle may place a small temporary hold.

## 2. Create the server

In the Oracle Cloud console, open **Compute → Instances → Create instance**.

| Setting | Choose |
| --- | --- |
| Name | `pocket-librarian` |
| Image | **Canonical Ubuntu 24.04** |
| Shape | **Ampere → VM.Standard.A1.Flex**, **2 OCPUs**, **12 GB memory** (labelled "Always Free-eligible") |
| Networking | Create a new virtual cloud network with a **public subnet**, and keep **Assign a public IPv4 address** on |
| SSH keys | **Generate a key pair for me**, then **download the private key** and keep it safe |
| Boot volume | Default (50 GB) |

Click **Create**. When the instance shows **Running**, copy its **Public IP address**.

> **"Out of capacity"?** Free Arm servers are popular. Try another *availability domain* in the same form, try again later, or use **1 OCPU / 6 GB**, which is enough for this app.

> In mid-2026 Oracle lowered the free Arm allowance to 2 OCPUs and 12 GB in total. If the form shows different limits, use what it marks as Always Free-eligible.

## 3. Open the web ports in Oracle's firewall

Oracle blocks incoming traffic by default.

1. On the instance page, click the **subnet** link (under *Primary VNIC*), then its **Default Security List**.
2. **Add Ingress Rules**, twice:
   - Source CIDR `0.0.0.0/0`, IP protocol **TCP**, destination port **80**
   - Source CIDR `0.0.0.0/0`, IP protocol **TCP**, destination port **443**

## 4. Connect to the server

On Windows, use PowerShell (OpenSSH is built in). Replace the key path and IP:

```powershell
ssh -i C:\path\to\ssh-key.key ubuntu@203.0.113.7
```

If SSH complains that the key file permissions are too open, run this once and try again:

```powershell
icacls C:\path\to\ssh-key.key /inheritance:r /grant:r "$($env:USERNAME):R"
```

Everything from here on is typed **on the server**.

## 5. Open the same ports in Ubuntu's firewall

Oracle's Ubuntu images also block ports in the server's own firewall:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 6. Install Docker

This uses Docker's official install script:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
exit
```

`exit` logs you out so the group change takes effect. Connect again with the same `ssh` command as in step 4, then check it works:

```bash
docker run --rm hello-world
```

## 7. Get the code

Commit and push your latest changes to GitHub from your laptop first. Then, on the server:

```bash
git clone https://github.com/naufalhakims/rag-short-project.git pocket-librarian
cd pocket-librarian
```

(If the repository is private, GitHub asks for your username and a personal access token instead of a password.)

## 8. Create the settings file

```bash
nano .env
```

Paste this, fill in your values, then save with `Ctrl+O`, `Enter`, `Ctrl+X`:

```env
GROQ_API_KEY=your-groq-key
GOOGLE_API_KEY=your-gemini-key        # optional backup model
APP_PASSWORD=choose-a-long-password   # required: the site is public
DOMAIN=203-0-113-7.sslip.io           # your public IP with dashes, + .sslip.io
```

**About `DOMAIN`:** HTTPS needs a domain name. [sslip.io](https://sslip.io) is a free service that turns any IP into one: for the IP `203.0.113.7`, use `203-0-113-7.sslip.io`. You can use your own domain instead; point its DNS `A` record at the server's IP.

Keep the file private:

```bash
chmod 600 .env
```

## 9. Start it

```bash
docker compose up -d --build
```

The first build takes about 5–10 minutes. It installs everything and downloads the search models into the image. Watch progress with:

```bash
docker compose logs -f
```

When you see `Pocket Librarian 2.0.0 is up` and Caddy reports `certificate obtained successfully`, press `Ctrl+C` (the app keeps running). Open `https://<your DOMAIN>` and log in with `APP_PASSWORD`.

The containers restart automatically after a crash or a server reboot.

---

## Everyday tasks

**Update to a new version** (after pushing changes to GitHub):

```bash
cd ~/pocket-librarian
git pull
docker compose up -d --build
```

Documents and chats are kept: they live in a Docker volume, not in the image.

**See what's happening:**

```bash
docker compose ps              # are both containers running?
docker compose logs -f app     # the app's log
```

**Back up your data:**

```bash
docker compose cp app:/home/user/data ./backup-$(date +%F)
```

Copy the backup folder to your laptop with `scp -r -i C:\path\to\ssh-key.key ubuntu@<ip>:pocket-librarian/backup-* .` (run that in PowerShell on the laptop).

**Bring your existing documents and chats from the laptop:** stop the app on the laptop, copy the `data` folder up, then on the server:

```bash
# on the laptop (PowerShell), from the project folder:
#   scp -r -i C:\path\to\ssh-key.key .\data\librarian.db .\data\chroma ubuntu@<ip>:pocket-librarian/import/
docker compose stop app
docker compose cp ./import/. app:/home/user/data/
docker compose run --rm --user root --entrypoint chown app -R 1000:1000 /home/user/data
docker compose start app
```

Use the same `EMBEDDING_MODEL` on both machines, or the app re-indexes everything on its first start (which is automatic, just slower).

**Change the password or keys:** edit `.env`, then `docker compose up -d`. Changing `APP_PASSWORD` logs everyone out.

---

## Troubleshooting

- **The site does not load at all.** Check both firewalls: the ingress rules for ports 80 and 443 (step 3) *and* the `iptables` commands (step 5). From the server, `curl -I http://localhost` should answer.
- **Certificate errors in the Caddy log.** Let's Encrypt must reach the server on port 80, so the firewalls above apply here too. Check that `DOMAIN` matches the IP exactly. If sslip.io hits Let's Encrypt's rate limits, get a free subdomain from [DuckDNS](https://www.duckdns.org) and use that instead.
- **"Out of capacity" when creating the server.** See step 2.
- **The build stops with a memory error.** Check the server really has the A1 shape with at least 6 GB.
- **Oracle stopped the server.** Oracle may reclaim Always Free servers that sit almost completely idle for a long time. Start it again from the console. Your data is on its disk.
