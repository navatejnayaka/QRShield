from flask import Flask, request, render_template_string
import cv2
import numpy as np
import re
import ipaddress
from urllib.parse import urlsplit, parse_qs, unquote
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# =========================================================
# SECURITY DATA
# =========================================================

SUSPICIOUS_WORDS = [
    "verify", "verification", "login", "signin", "sign-in",
    "urgent", "claim", "reward", "free", "password",
    "payment", "wallet", "update", "security", "confirm",
    "bank", "refund", "bonus", "gift", "prize", "otp", "kyc"
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd",
    "ow.ly", "buff.ly", "cutt.ly", "shorturl.at",
    "rb.gy", "rebrand.ly", "tiny.cc", "lnkd.in"
}

SUSPICIOUS_TLDS = {
    "zip", "top", "click", "gq", "tk", "ml", "ga",
    "cf", "work", "country"
}

FEATURE_NAMES = [
    "https",
    "ip_host",
    "at_symbol",
    "credentials",
    "punycode",
    "many_subdomains",
    "long_url",
    "long_path",
    "suspicious_words",
    "query_params",
    "nonstandard_port",
    "percent_encoded",
    "url_shortener",
    "suspicious_tld",
    "unicode_host",
    "long_host",
    "many_hyphens"
]


# =========================================================
# SYNTHETIC ML MODEL
# =========================================================

def create_model():

    rng = np.random.default_rng(42)
    rows = []

    weights = np.array([
        -8, 25, 20, 20, 25, 12, 8, 8, 15,
        4, 10, 5, 15, 8, 15, 8, 5
    ])

    for _ in range(1500):

        features = rng.integers(
            0, 2, size=len(FEATURE_NAMES)
        )

        features[0] = rng.choice(
            [0, 1],
            p=[0.2, 0.8]
        )

        score = float(np.dot(features, weights))
        score += rng.normal(0, 7)

        label = int(score >= 32)

        rows.append(list(features) + [label])

    model_data = np.array(rows)

    X = model_data[:, :-1]
    y = model_data[:, -1]

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )

    model.fit(X, y)

    return model


model = create_model()


# =========================================================
# URL ANALYSIS
# =========================================================

def analyze_url(raw_url):

    value = raw_url.strip()

    if not value:
        return {
            "valid": False,
            "error": "Empty URL."
        }

    # Add HTTPS if user gives only a domain
    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        value
    ):
        value = "https://" + value

    try:
        parsed = urlsplit(value)
    except ValueError:
        return {
            "valid": False,
            "error": "URL could not be parsed."
        }

    if parsed.scheme.lower() not in ["http", "https"]:
        return {
            "valid": False,
            "error": "Only HTTP and HTTPS links are currently analyzed."
        }

    if not parsed.hostname:
        return {
            "valid": False,
            "error": "No domain/hostname found."
        }

    host = parsed.hostname.lower()

    # IP detection
    try:
        is_ip = 1
        ipaddress.ip_address(host)
    except ValueError:
        is_ip = 0

    try:
        port = parsed.port
    except ValueError:
        return {
            "valid": False,
            "error": "Invalid port."
        }

    tld = (
        host.rsplit(".", 1)[-1]
        if "." in host else ""
    )

    word_hits = [
        word for word in SUSPICIOUS_WORDS
        if word in value.lower()
    ]

    features = {
        "https": int(parsed.scheme.lower() == "https"),
        "ip_host": is_ip,
        "at_symbol": int("@" in value),
        "credentials": int(
            parsed.username is not None
            or parsed.password is not None
        ),
        "punycode": int("xn--" in host),
        "many_subdomains": int(host.count(".") >= 3),
        "long_url": int(len(value) > 120),
        "long_path": int(len(parsed.path) > 80),
        "suspicious_words": int(bool(word_hits)),
        "query_params": int(bool(parsed.query)),
        "nonstandard_port": int(
            port is not None
            and not (
                (parsed.scheme.lower() == "http" and port == 80)
                or
                (parsed.scheme.lower() == "https" and port == 443)
            )
        ),
        "percent_encoded": int("%" in value),
        "url_shortener": int(
            host in SHORTENERS
            or any(host.endswith("." + x) for x in SHORTENERS)
        ),
        "suspicious_tld": int(
            tld in SUSPICIOUS_TLDS
        ),
        "unicode_host": int(
            any(ord(ch) > 127 for ch in host)
        ),
        "long_host": int(len(host) > 45),
        "many_hyphens": int(host.count("-") >= 3)
    }

    feature_vector = np.array([[
        features[name]
        for name in FEATURE_NAMES
    ]])

    ai_probability = float(
        model.predict_proba(feature_vector)[0][1]
    )

    # Explicit security scoring
    weights = {
        "https": -8,
        "ip_host": 25,
        "at_symbol": 20,
        "credentials": 20,
        "punycode": 25,
        "many_subdomains": 12,
        "long_url": 8,
        "long_path": 8,
        "suspicious_words": 15,
        "query_params": 4,
        "nonstandard_port": 10,
        "percent_encoded": 5,
        "url_shortener": 15,
        "suspicious_tld": 8,
        "unicode_host": 15,
        "long_host": 8,
        "many_hyphens": 5
    }

    points = sum(
        weights[name]
        for name in weights
        if features[name]
    )

    rule_score = max(
        0,
        min(100, points)
    )

    final_score = (
        rule_score * 0.55
        +
        ai_probability * 100 * 0.45
    )

    reasons = []

    if not features["https"]:
        reasons.append("Connection does not use HTTPS.")

    if features["ip_host"]:
        reasons.append(
            "Destination uses an IP address instead of a normal domain."
        )

    if features["at_symbol"]:
        reasons.append("URL contains an @ symbol.")

    if features["credentials"]:
        reasons.append(
            "URL contains embedded username/password information."
        )

    if features["punycode"]:
        reasons.append("Domain contains punycode.")

    if features["many_subdomains"]:
        reasons.append("Domain contains many subdomains.")

    if features["long_url"]:
        reasons.append("URL is unusually long.")

    if features["long_path"]:
        reasons.append("URL path is unusually long.")

    if features["suspicious_words"]:
        reasons.append(
            "Suspicious-looking keywords detected: "
            + ", ".join(word_hits[:5])
        )

    if features["query_params"]:
        reasons.append("URL contains query parameters.")

    if features["nonstandard_port"]:
        reasons.append(
            f"Non-standard port detected: {port}."
        )

    if features["percent_encoded"]:
        reasons.append(
            "URL contains percent-encoded characters."
        )

    if features["url_shortener"]:
        reasons.append(
            "URL uses a known URL-shortening service."
        )

    if features["suspicious_tld"]:
        reasons.append(
            f"Domain uses .{tld}, which is treated as a risk signal."
        )

    if features["unicode_host"]:
        reasons.append(
            "Domain contains non-ASCII characters."
        )

    if features["long_host"]:
        reasons.append(
            "Hostname is unusually long."
        )

    if features["many_hyphens"]:
        reasons.append(
            "Hostname contains many hyphens."
        )

    if final_score >= 65:
        risk = "HIGH RISK"
    elif final_score >= 35:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    if risk == "HIGH RISK":
        action = (
            "Do not open or pay through this destination "
            "until it has been independently verified."
        )
    elif risk == "MEDIUM RISK":
        action = (
            "Verify the destination, merchant and payment "
            "details before continuing."
        )
    else:
        action = (
            "No strong suspicious indicators were detected. "
            "Still verify the destination before payment."
        )

    return {
        "valid": True,
        "risk": risk,
        "score": final_score,
        "ai_probability": ai_probability,
        "domain": host,
        "url": value,
        "reasons": reasons,
        "action": action
    }


# =========================================================
# QR DECODER
# =========================================================

def decode_qr(file_bytes):

    try:

        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return ""

        detector = cv2.QRCodeDetector()

        data, _, _ = detector.detectAndDecode(image)

        if data:
            return data.strip()

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        data, _, _ = detector.detectAndDecode(gray)

        if data:
            return data.strip()

        return ""

    except Exception:

        return ""


# =========================================================
# QR TYPE DETECTION
# =========================================================

def detect_qr_type(data):

    lower = data.strip().lower()

    if lower.startswith("upi://pay"):
        return "UPI PAYMENT"

    if lower.startswith("http://") or lower.startswith("https://"):
        return "WEB URL"

    if lower.startswith("mailto:"):
        return "EMAIL"

    if lower.startswith("tel:"):
        return "PHONE"

    if lower.startswith("wifi:"):
        return "WI-FI"

    if "http://" in lower or "https://" in lower:
        return "TEXT WITH URL"

    return "TEXT / OTHER"


# =========================================================
# HTML
# =========================================================

HTML = """
<!DOCTYPE html>
<html>
<head>

    <title>QRShield</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #0b1220;
            color: #f8fafc;
        }

        .container {
            width: 92%;
            max-width: 950px;
            margin: 40px auto;
        }

        .hero {
            text-align: center;
            padding: 30px;
        }

        .hero h1 {
            font-size: 52px;
            margin: 0;
        }

        .hero h2 {
            margin: 8px 0;
            font-size: 22px;
        }

        .hero p {
            opacity: 0.75;
        }

        .card {
            background: #111827;
            border: 1px solid #263244;
            border-radius: 18px;
            padding: 24px;
            margin-top: 20px;
        }

        input[type=text],
        input[type=file] {
            width: 100%;
            padding: 14px;
            margin-top: 10px;
            border-radius: 10px;
            border: 1px solid #334155;
            background: #0f172a;
            color: white;
        }

        button {
            width: 100%;
            padding: 14px;
            margin-top: 15px;
            border: none;
            border-radius: 10px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            opacity: 0.9;
        }

        .risk {
            padding: 16px;
            border-radius: 12px;
            margin-top: 18px;
            font-size: 22px;
            font-weight: bold;
        }

        .high {
            background: #451a1a;
            color: #fecaca;
        }

        .medium {
            background: #422006;
            color: #fed7aa;
        }

        .low {
            background: #052e1b;
            color: #bbf7d0;
        }

        .signal {
            padding: 10px 0;
            border-bottom: 1px solid #243047;
        }

        .metric {
            display: inline-block;
            width: 48%;
            padding: 15px;
            background: #0f172a;
            border-radius: 10px;
            margin-top: 15px;
        }

        .warning {
            background: #172554;
            padding: 15px;
            border-radius: 10px;
            margin-top: 18px;
        }

        .success {
            background: #052e1b;
            padding: 15px;
            border-radius: 10px;
            margin-top: 18px;
        }

        .footer {
            text-align: center;
            opacity: 0.6;
            margin: 30px;
        }

    </style>

</head>

<body>

<div class="container">

    <div class="hero">

        <h1>🛡️ QRShield</h1>

        <h2>The Payment Trap Investigator</h2>

        <p>
            Detect • Explain • Assist
        </p>

        <p>
            Analyze any decodable QR or URL for suspicious
            security characteristics.
        </p>

    </div>


    <div class="card">

        <h2>🔗 Analyze Any URL or Domain</h2>

        <form method="POST">

            <input
                type="hidden"
                name="action"
                value="url"
            >

            <input
                type="text"
                name="url"
                placeholder="https://example.com or example.com"
                required
            >

            <button type="submit">
                🔍 Analyze Link
            </button>

        </form>

    </div>


    <div class="card">

        <h2>📷 Analyze Any QR</h2>

        <form method="POST" enctype="multipart/form-data">

            <input
                type="hidden"
                name="action"
                value="qr"
            >

            <input
                type="file"
                name="qr"
                accept=".png,.jpg,.jpeg"
                required
            >

            <button type="submit">
                📷 Decode & Analyze QR
            </button>

        </form>

    </div>


    {% if result %}

    <div class="card">

        {% if result.error %}

            <div class="warning">
                ⚠️ {{ result.error }}
            </div>

        {% else %}

            {% if result.risk == "HIGH RISK" %}

                <div class="risk high">
                    🚨 HIGH RISK
                </div>

            {% elif result.risk == "MEDIUM RISK" %}

                <div class="risk medium">
                    ⚠️ MEDIUM RISK
                </div>

            {% else %}

                <div class="risk low">
                    ✅ LOW RISK
                </div>

            {% endif %}


            <div class="metric">

                <b>AI-Assisted Risk Score</b>

                <br>

                {{ "%.1f"|format(result.score) }}%

            </div>


            <div class="metric">

                <b>AI Estimate</b>

                <br>

                {{ "%.1f"|format(result.ai_probability * 100) }}%

            </div>


            <h3>🌐 Domain</h3>

            <p>
                {{ result.domain }}
            </p>


            <h3>🔗 Analyzed URL</h3>

            <p style="word-break: break-all;">
                {{ result.url }}
            </p>


            <h3>🔎 Detected Security Signals</h3>

            {% if result.reasons %}

                {% for reason in result.reasons %}

                    <div class="signal">
                        ⚠️ {{ reason }}
                    </div>

                {% endfor %}

            {% else %}

                <div class="signal">
                    ✅ No obvious suspicious signals detected.
                </div>

            {% endif %}


            <div class="warning">

                <b>🛡️ Recommended Action</b>

                <br><br>

                {{ result.action }}

            </div>


            <div class="warning">

                ℹ️ QRShield performs static feature-based
                analysis. It does not open the destination and
                cannot guarantee that a URL is safe.

            </div>

        {% endif %}

    </div>

    {% endif %}


    {% if qr_result %}

    <div class="card">

        <h2>📦 QR Analysis Result</h2>

        <p>
            <b>Decoded Content:</b>
        </p>

        <p style="word-break: break-all;">
            {{ qr_result.content }}
        </p>

        <p>
            <b>Detected Type:</b>
            {{ qr_result.qr_type }}
        </p>


        {% if qr_result.web_result %}

            {% set result = qr_result.web_result %}

            {% if result.risk == "HIGH RISK" %}

                <div class="risk high">
                    🚨 HIGH RISK
                </div>

            {% elif result.risk == "MEDIUM RISK" %}

                <div class="risk medium">
                    ⚠️ MEDIUM RISK
                </div>

            {% else %}

                <div class="risk low">
                    ✅ LOW RISK
                </div>

            {% endif %}


            <div class="metric">

                <b>Risk Score</b>

                <br>

                {{ "%.1f"|format(result.score) }}%

            </div>


            <h3>🌐 Domain</h3>

            <p>
                {{ result.domain }}
            </p>


            <h3>🔎 Security Signals</h3>

            {% if result.reasons %}

                {% for reason in result.reasons %}

                    <div class="signal">
                        ⚠️ {{ reason }}
                    </div>

                {% endfor %}

            {% else %}

                <div class="signal">
                    ✅ No obvious suspicious signals detected.
                </div>

            {% endif %}


            <div class="warning">

                <b>🛡️ Recommended Action</b>

                <br><br>

                {{ result.action }}

            </div>

        {% elif qr_result.message %}

            <div class="success">
                ✅ {{ qr_result.message }}
            </div>

        {% endif %}

    </div>

    {% endif %}


    <div class="footer">

        🛡️ QRShield • AI × Cybersecurity Mini Hackathon 2026

        <br>

        Prototype using synthetic demonstration data.

    </div>

</div>

</body>
</html>
"""


# =========================================================
# ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    qr_result = None

    if request.method == "POST":

        action = request.form.get("action")

        # -------------------------------
        # URL
        # -------------------------------

        if action == "url":

            url = request.form.get(
                "url",
                ""
            )

            result = analyze_url(url)

        # -------------------------------
        # QR
        # -------------------------------

        elif action == "qr":

            uploaded = request.files.get(
                "qr"
            )

            if uploaded:

                payload = decode_qr(
                    uploaded.read()
                )

                if not payload:

                    qr_result = {
                        "content": "",
                        "qr_type": "Unknown",
                        "message": (
                            "Could not decode this QR. "
                            "Try a clear image with the complete "
                            "QR visible."
                        )
                    }

                else:

                    qr_type = detect_qr_type(
                        payload
                    )

                    web_result = None
                    message = None

                    if qr_type == "WEB URL":

                        web_result = analyze_url(
                            payload
                        )

                    elif qr_type == "TEXT WITH URL":

                        found_urls = re.findall(
                            r"https?://[^\s]+",
                            payload
                        )

                        if found_urls:

                            web_result = analyze_url(
                                found_urls[0]
                            )

                    elif qr_type == "UPI PAYMENT":

                        message = (
                            "UPI payment payload detected. "
                            "The QR was decoded successfully. "
                            "Verify the payee and amount before paying."
                        )

                    elif qr_type == "EMAIL":

                        message = (
                            "Email QR detected. "
                            "The payload was decoded successfully."
                        )

                    elif qr_type == "PHONE":

                        message = (
                            "Phone QR detected. "
                            "The payload was decoded successfully."
                        )

                    elif qr_type == "WI-FI":

                        message = (
                            "Wi-Fi QR detected. "
                            "The payload was decoded successfully."
                        )

                    else:

                        message = (
                            "QR decoded successfully. "
                            "The payload is not a web URL."
                        )

                    qr_result = {
                        "content": payload,
                        "qr_type": qr_type,
                        "web_result": web_result,
                        "message": message
                    }

    return render_template_string(
        HTML,
        result=result,
        qr_result=qr_result
    )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )