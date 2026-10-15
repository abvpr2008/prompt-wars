# ChronoPulse AI — Security Policy & Responsible AI Guardrails 🛡️

## 1. Overview & Security Philosophy
**ChronoPulse AI** is engineered as a Class II Software-as-a-Medical-Device (SaMD) clinical decision support prototype. Because clinical AI systems handle sensitive health metrics and potential Protected Health Information (PHI), security, privacy, and determinism are primary design requirements.

---

## 2. Secrets Management & Environment Hygiene
- **Zero Secrets in Git:** All API keys, cloud provider tokens, and confidential keys must reside exclusively in local `.env` files.
- **Gitignore Enforcement:** `.gitignore` blocks `.env`, `.env.*`, certificates (`.pem`, `.key`, `.cert`), credential JSONs (`*credentials*.json`), and cache files.
- **Environment Template:** Developers and evaluators should consult [`.env.example`](file:///.env.example) to configure environment variables.
- **Automated Fallback Secret Key:** If `SECRET_KEY` is not provided in `.env`, the server generates a cryptographically secure 256-bit random hex key (`secrets.token_hex(32)`) on boot rather than falling back to a static string.

---

## 3. Network & Transport Security
- **Production Loopback Binding:** The server defaults to `127.0.0.1` rather than `0.0.0.0` unless explicitly overridden via `HOST` in production deployments.
- **Debug Mode Lockdown:** `FLASK_DEBUG` defaults to `false`. Running debug mode on public interfaces is strictly disabled.
- **Configurable CORS:** Cross-Origin Resource Sharing is locked down to a whitelist (`ALLOWED_ORIGINS`) rather than wildcard (`*`) access.
- **Strict HTTP Security Headers:** Every HTTP response includes:
  - `Content-Security-Policy (CSP)`: Whitelists trusted CDNs (Tailwind, Lucide, Chart.js, Google Fonts) and restricts unsafe frame ancestors.
  - `X-Content-Type-Options: nosniff`: Prevents MIME-type sniffing.
  - `X-Frame-Options: SAMEORIGIN`: Prevents clickjacking attacks.
  - `X-XSS-Protection: 1; mode=block`: Activates browser XSS filtering.
  - `Referrer-Policy: strict-origin-when-cross-origin`: Minimizes referrer leakage.
  - `Permissions-Policy`: Disables camera, microphone, and geolocation access unless explicitly needed.

---

## 4. Input Sanitization & Anti-DoS Guardrails
- **File Upload Limits:** Configured with `MAX_CONTENT_LENGTH = 16 MB` (`16777216` bytes) to prevent memory exhaustion and buffer overflow DoS attacks.
- **File Extension Whitelist:** OCR document ingestion restricts uploads to approved medical file types: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`.
- **Filename Sanitization:** All incoming filenames are sanitized through `werkzeug.utils.secure_filename` before processing.
- **Path Traversal Protection:** Static asset delivery verifies that canonical filesystem paths are strictly contained within the `assets/` directory.
- **Numerical Clamping:** Clinical biomarkers and colorimetric coordinates are bounded to physiological sanity limits to prevent division by zero, `NaN`, or floating-point overflow (`sigmoid` overflow protection).

---

## 5. HIPAA & Patient Privacy (PII Scrubber)
- **Safe-Harbor De-Identification:** Built-in client-side and backend de-identification engines strip 18 direct HIPAA identifiers.
- **Masked Session Pseudonyms:** Patient records are referenced by pseudonymized tokens (e.g. `Token #CP-8841`) during clinical reviews.
- **XSS Escaping:** Dynamic text rendering across demographic banners and clinical triage summaries is sanitized through HTML entity encoding.

---

## 6. Responsible AI & Clinical Safety Fallback
- **ISO-14971 Medical Risk Alignment:** The prediction pipeline monitors model confidence and slope velocity.
- **Uncertainty Fallback Protocol:** If prediction confidence drops below 75% or velocity metrics exceed two standard deviations of physiological drift:
  - Autonomous inference confidence is suppressed.
  - An active clinical fallback banner (`ISO-14971-CP403`) is triggered.
  - The system mandates specialist verification before action plans are rendered.

---

## 7. Reporting Security Vulnerabilities
If you discover a security vulnerability or responsible AI flaw, please submit an issue or reach out to the project maintainers directly.
