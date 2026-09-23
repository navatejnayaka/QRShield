from flask import Flask, request, render_template_string
import cv2
import numpy as np
import re
from urllib.parse import urlparse, parse_qs
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


# =========================================================
# AI / DEMO MODEL
# =========================================================

def create_demo_model():
    rows = []

    for _ in range(1000):
        https = np.random.randint(0, 2)
        long_url = np.random.randint(0, 2)
        has_at = np.random.randint(0, 2)
        has_ip = np.random.randint(0, 2)
        punycode = np.random.randint(0, 2)
        many_subdomains = np.random.randint(0, 2)
        suspicious_words = np.random.randint(0, 2)
        query = np.random.randint(0, 2)
        long_path = np.random.randint(0, 2)
        shortener = np.random.randint(0, 2)
        unusual_port = np.random.randint(0, 2)

        risk_points = (
            (1 - https) * 1
            + long_url * 1
            + has_at * 3
            + has_ip * 3
            + punycode * 3
            + many_subdomains * 2
            + suspicious_words * 2
            + query * 1
            + long_path * 1
            + shortener * 2
            + unusual_port * 2
        )

        label = int(risk_points >= 5)

        rows.append([
            https,
            long_url,
            has_at,
            has_ip,
            punycode,
            many_subdomains,
            suspicious_words,
            query,
            long_path,
            shortener,
            unusual_port,
            label
        ])

    data = np.array(rows)

    X = data[:, :-1]
    y = data[:, -1]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    return model


model = create_demo_model()


# =========================================================
# URL FEATURE EXTRACTION
# =========================================================

def extract_features(url):

    url_lower = url.lower().strip()

    parsed = urlparse(url_lower)

    hostname = parsed.hostname or ""

    suspicious_words_list = [
        "verify",
        "login",
        "urgent",
        "claim",
        "reward",
        "free",
        "password",
        "payment",
        "wallet",
        "update",
        "account",
        "confirm",
        "security",
        "refund"
    ]

    shortener_domains = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "is.gd",
        "cutt.ly",
        "rb.gy",
        "shorturl.at"
    ]

    features = {
        "https": int(url_lower.startswith("https://")),

        "long_url": int(len(url) > 100),

        "has_at": int("@" in url),

        "has_ip": int(bool(
            re.search(
                r"https?://\d{1,3}(\.\d{1,3}){3}",
                url_lower
            )
        )),

        "punycode": int(
            "xn--" in hostname
        ),

        "many_subdomains": int(
            hostname.count(".") >= 4
        ),

        "suspicious_words": int(
            any(word in url_lower for word in suspicious_words_list)
        ),

        "query": int(
            bool(parsed.query)
        ),

        "long_path": int(
            len(parsed.path) > 80
        ),

        "shortener": int(
            any(
                hostname == domain or
                hostname.endswith("." + domain)
                for domain in shortener_domains
            )
        ),

        "unusual_port": int(
            parsed.port is not None and
            parsed.port not in [80, 443]
        )
    }

    return features


# =========================================================
# URL ANALYSIS
# =========================================================

def analyze_url(raw_url):

    url = raw_url.strip()

    if not url:
        return {
            "level": "UNKNOWN",
            "score": 0,
            "ai_probability": 0,
            "domain": "",
            "url": "",
            "reasons": [],
            "action": "Enter a URL to analyze."
        }

    # Add scheme if user enters only a domain
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url

    features = extract_features(url)

    values = np.array([
        list(features.values())
    ])

    ai_probability = float(
        model.predict_proba(values)[0][1]
    )

    reasons = []

    if not features["https"]:
        reasons.append(
            "Connection does not use HTTPS."
        )

    if features["long_url"]:
        reasons.append(
            "URL is unusually long."
        )

    if features["has_at"]:
        reasons.append(
            "URL contains an @ symbol."
        )

    if features["has_ip"]:
        reasons.append(
            "Destination uses an IP address instead of a normal domain."
        )

    if features["punycode"]:
        reasons.append(
            "Domain contains punycode, which can be used for look-alike domains."
        )

    if features["many_subdomains"]:
        reasons.append(
            "URL contains many subdomains."
        )

    if features["suspicious_words"]:
        reasons.append(
            "URL contains words commonly associated with social-engineering attempts."
        )

    if features["query"]:
        reasons.append(
            "URL contains query parameters."
        )

    if features["long_path"]:
        reasons.append(
            "URL contains an unusually long path."
        )

    if features["shortener"]:
        reasons.append(
            "URL uses a known URL-shortening service."
        )

    if features["unusual_port"]:
        reasons.append(
            "URL uses an unusual network port."
        )

    signal_count = len(reasons)

    # Rule-based score
    rule_points = (
        (1 - features["https"]) * 8
        + features["long_url"] * 7
        + features["has_at"] * 20
        + features["has_ip"] * 20
        + features["punycode"] * 20
        + features["many_subdomains"] * 12
        + features["suspicious_words"] * 12
        + features["query"] * 5
        + features["long_path"] * 6
        + features["shortener"] * 10
        + features["unusual_port"] * 10
    )

    rule_score = min(rule_points, 100)

    # Combine static rules + ML estimate
    final_score = (
        rule_score * 0.65
        + ai_probability * 100 * 0.35
    )

    final_score = round(
        min(max(final_score, 0), 100),
        1
    )

    if final_score >= 65 or signal_count >= 5:
        level = "HIGH RISK"

    elif final_score >= 30 or signal_count >= 2:
        level = "MEDIUM RISK"

    else:
        level = "LOW RISK"

    parsed = urlparse(url)

    domain = parsed.hostname or "Unknown"

    if level == "HIGH RISK":
        action = (
            "Do not proceed without independent verification. "
            "Avoid entering passwords, OTPs or payment credentials."
        )

    elif level == "MEDIUM RISK":
        action = (
            "Verify the destination and recipient independently "
            "before continuing."
        )

    else:
        action = (
            "No strong suspicious indicators were detected. "
            "Still verify the destination before payment or login."
        )

    return {
        "level": level,
        "score": final_score,
        "ai_probability": round(ai_probability * 100, 1),
        "domain": domain,
        "url": url,
        "reasons": reasons,
        "action": action
    }


# =========================================================
# QR DECODER
# =========================================================

def decode_qr(file_bytes):

    try:

        # Convert uploaded bytes into an OpenCV image
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

        # -------------------------------------------------
        # Attempt 1: Original image
        # -------------------------------------------------

        data, points, _ = detector.detectAndDecode(image)

        if data:
            return data.strip()

        # -------------------------------------------------
        # Attempt 2: Multiple QR codes
        # -------------------------------------------------

        try:

            ok, decoded_info, points, _ = (
                detector.detectAndDecodeMulti(image)
            )

            if ok and decoded_info:

                for item in decoded_info:

                    if item and item.strip():
                        return item.strip()

        except Exception:
            pass

        # -------------------------------------------------
        # Prepare grayscale
        # -------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # -------------------------------------------------
        # Attempt 3: Enlarged images
        # -------------------------------------------------

        processed_images = []

        for scale in [2, 3, 4]:

            enlarged = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            processed_images.append(enlarged)

        # -------------------------------------------------
        # Attempt 4: Threshold
        # -------------------------------------------------

        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        processed_images.append(threshold)

        # -------------------------------------------------
        # Attempt 5: Adaptive threshold
        # -------------------------------------------------

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5
        )

        processed_images.append(adaptive)

        # -------------------------------------------------
        # Try all processed images
        # -------------------------------------------------

        for processed in processed_images:

            data, points, _ = (
                detector.detectAndDecode(processed)
            )

            if data:
                return data.strip()

            try:

                ok, decoded_info, points, _ = (
                    detector.detectAndDecodeMulti(processed)
                )

                if ok and decoded_info:

                    for item in decoded_info:

                        if item and item.strip():
                            return item.strip()

            except Exception:
                pass

        # -------------------------------------------------
        # Attempt 6: Inverted image
        # -------------------------------------------------

        inverted = cv2.bitwise_not(gray)

        data, points, _ = (
            detector.detectAndDecode(inverted)
        )

        if data:
            return data.strip()

        return ""

    except Exception as e:

        print("QR decode error:", e)

        return ""


# =========================================================
# QR TYPE DETECTION
# =========================================================

def detect_qr_type(content):

    text = content.strip()

    lower = text.lower()

    if lower.startswith("upi://pay"):
        return "UPI PAYMENT"

    if lower.startswith("http://") or \
       lower.startswith("https://"):

        return "WEB URL"

    if lower.startswith("mailto:"):
        return "EMAIL"

    if lower.startswith("tel:"):
        return "PHONE"

    if lower.startswith("wifi:"):
        return "WI-FI"

    if "begin:vcard" in lower:
        return "CONTACT"

    return "TEXT / OTHER"


# =========================================================
# UPI ANALYSIS
# =========================================================

def analyze_upi(content):

    try:

        parsed = urlparse(content)

        params = parse_qs(
            parsed.query
        )

        payee = params.get(
            "pa",
            [None]
        )[0]

        name = params.get(
            "pn",
            [None]
        )[0]

        amount = params.get(
            "am",
            [None]
        )[0]

        currency = params.get(
            "cu",
            ["INR"]
        )[0]

        reasons = []

        if not payee:
            reasons.append(
                "No payee UPI ID was found in the payload."
            )

        else:
            if "@" not in payee:
                reasons.append(
                    "The payee identifier does not follow a typical UPI ID format."
                )

        if not amount:
            amount_text = "Not specified"
        else:
            amount_text = amount + " " + currency

        action = (
            "Verify the payee name and UPI ID independently "
            "before making any payment."
        )

        return {
            "payee": payee or "Not specified",
            "name": name or "Not specified",
            "amount": amount_text,
            "reasons": reasons,
            "action": action
        }

    except Exception:

        return {
            "payee": "Unable to parse",
            "name": "Unable to parse",
            "amount": "Unable to parse",
            "reasons": [
                "The UPI payload could not be fully parsed."
            ],
            "action": (
                "Verify the payment details independently "
                "before continuing."
            )
        }


# =========================================================
# HTML
# =========================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>QRShield</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #07101f;
    color: #f5f7fb;
}

.container {
    width: 92%;
    max-width: 1000px;
    margin: auto;
    padding: 35px 0 50px;
}

.header {
    text-align: center;
    margin-bottom: 35px;
}

.logo {
    font-size: 48px;
    font-weight: 800;
}

.subtitle {
    font-size: 21px;
    font-weight: bold;
    margin-top: 5px;
}

.tagline {
    color: #9ca9bc;
    margin-top: 10px;
}

.card {
    background: #101a2c;
    border: 1px solid #25334a;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 25px;
}

h2 {
    margin-top: 0;
}

input[type=text] {
    width: 100%;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #35445e;
    background: #0b1424;
    color: white;
    font-size: 16px;
    margin: 12px 0;
}

input[type=file] {
    width: 100%;
    padding: 15px;
    background: #0b1424;
    border-radius: 10px;
    color: white;
    margin: 12px 0;
}

button {
    width: 100%;
    padding: 15px;
    border: 0;
    border-radius: 10px;
    background: #2563eb;
    color: white;
    font-size: 17px;
    font-weight: bold;
    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.result {
    margin-top: 20px;
}

.risk {
    padding: 18px;
    border-radius: 12px;
    font-size: 24px;
    font-weight: bold;
    margin: 15px 0;
}

.high {
    background: #45151a;
    border: 1px solid #ef4444;
}

.medium {
    background: #45340e;
    border: 1px solid #f59e0b;
}

.low {
    background: #073b29;
    border: 1px solid #22c55e;
}

.info {
    background: #10243d;
    border: 1px solid #31557d;
    padding: 16px;
    border-radius: 10px;
    margin-top: 15px;
}

.warning {
    background: #33260b;
    border: 1px solid #d99b16;
    padding: 16px;
    border-radius: 10px;
    margin-top: 15px;
}

.success {
    background: #073b29;
    border: 1px solid #22c55e;
    padding: 16px;
    border-radius: 10px;
    margin-top: 15px;
}

.signal {
    padding: 8px 0;
}

.open-button {
    display: inline-block;
    padding: 12px 20px;
    background: #2563eb;
    color: white;
    text-decoration: none;
    border-radius: 8px;
    font-weight: bold;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: #7f8da3;
    margin-top: 35px;
}

.small {
    color: #9ca9bc;
    font-size: 14px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}

td {
    padding: 10px;
    border-bottom: 1px solid #27354c;
}

td:first-child {
    font-weight: bold;
    width: 35%;
}

code {
    word-break: break-all;
}

</style>

</head>


<body>

<div class="container">

<div class="header">

<div class="logo">
🛡️ QRShield
</div>

<div class="subtitle">
The Payment Trap Investigator
</div>

<div class="tagline">
Detect • Explain • Assist
</div>

</div>


<!-- URL SECTION -->

<div class="card">

<h2>🔗 Analyze Any URL or Domain</h2>

<p class="small">
Analyze a website or payment destination for suspicious
security characteristics.
</p>

<form method="POST">

<input
    type="text"
    name="url"
    placeholder="https://example.com"
    value="{{ entered_url }}"
>

<button type="submit" name="action" value="url">
🔍 Analyze URL
</button>

</form>

</div>


<!-- QR SECTION -->

<div class="card">

<h2>📷 Analyze Any QR</h2>

<p class="small">
Upload a QR image. QRShield will decode the payload,
identify its type and analyze the destination when applicable.
</p>

<form method="POST"
      enctype="multipart/form-data">

<input
    type="file"
    name="qr_file"
    accept=".png,.jpg,.jpeg"
    required
>

<button type="submit" name="action" value="qr">
📷 Decode & Analyze QR
</button>

</form>

</div>


{% if url_result %}

<div class="card">

<h2>🛡️ QRShield Security Assessment</h2>

<div class="risk
{% if url_result.level == 'HIGH RISK' %}
high
{% elif url_result.level == 'MEDIUM RISK' %}
medium
{% else %}
low
{% endif %}
">

{% if url_result.level == 'HIGH RISK' %}
🚨
{% elif url_result.level == 'MEDIUM RISK' %}
⚠️
{% else %}
✅
{% endif %}

{{ url_result.level }}

</div>


<table>

<tr>
<td>AI-Assisted Risk Score</td>
<td>{{ url_result.score }}%</td>
</tr>

<tr>
<td>AI Estimate</td>
<td>{{ url_result.ai_probability }}%</td>
</tr>

<tr>
<td>Domain</td>
<td>{{ url_result.domain }}</td>
</tr>

<tr>
<td>Analyzed URL</td>
<td>
<code>{{ url_result.url }}</code>
</td>
</tr>

</table>


<h3>🔎 Detected Security Signals</h3>

{% if url_result.reasons %}

{% for reason in url_result.reasons %}

<div class="signal">
• {{ reason }}
</div>

{% endfor %}

{% else %}

<div class="success">
✅ No obvious suspicious signals detected.
</div>

{% endif %}


<div class="info">

<strong>Recommended Action</strong>

<br><br>

{{ url_result.action }}

</div>


<p>

<a
    class="open-button"
    href="{{ url_result.url }}"
    target="_blank"
    rel="noopener noreferrer"
>

🔗 Open Destination

</a>

</p>


<p class="small">
QRShield performs static feature-based analysis.
It does not guarantee that a destination is completely safe.
</p>

</div>

{% endif %}


{% if qr_result %}

<div class="card">

<h2>📦 QR Analysis Result</h2>

<h3>Decoded Content:</h3>

<div class="info">
<code>{{ qr_result.content }}</code>
</div>

<h3>
Detected Type:
{{ qr_result.type }}
</h3>


{% if qr_result.type == "WEB URL" %}

{% if qr_result.url_result %}

<div class="risk
{% if qr_result.url_result.level == 'HIGH RISK' %}
high
{% elif qr_result.url_result.level == 'MEDIUM RISK' %}
medium
{% else %}
low
{% endif %}
">

{% if qr_result.url_result.level == 'HIGH RISK' %}
🚨
{% elif qr_result.url_result.level == 'MEDIUM RISK' %}
⚠️
{% else %}
✅
{% endif %}

{{ qr_result.url_result.level }}

</div>


<table>

<tr>
<td>AI-Assisted Risk Score</td>
<td>{{ qr_result.url_result.score }}%</td>
</tr>

<tr>
<td>AI Estimate</td>
<td>{{ qr_result.url_result.ai_probability }}%</td>
</tr>

<tr>
<td>Domain</td>
<td>{{ qr_result.url_result.domain }}</td>
</tr>

</table>


<h3>🔎 Security Signals</h3>

{% if qr_result.url_result.reasons %}

{% for reason in qr_result.url_result.reasons %}

<div class="signal">
• {{ reason }}
</div>

{% endfor %}

{% else %}

<div class="success">
✅ No obvious suspicious signals detected.
</div>

{% endif %}


<div class="info">

<strong>Recommended Action</strong>

<br><br>

{{ qr_result.url_result.action }}

</div>


<a
    class="open-button"
    href="{{ qr_result.url_result.url }}"
    target="_blank"
    rel="noopener noreferrer"
>

🔗 Open Destination

</a>

{% endif %}


{% elif qr_result.type == "UPI PAYMENT" %}

<div class="warning">

<strong>💳 UPI PAYMENT DETECTED</strong>

</div>


<table>

<tr>
<td>Payee UPI ID</td>
<td>{{ qr_result.upi.payee }}</td>
</tr>

<tr>
<td>Payee Name</td>
<td>{{ qr_result.upi.name }}</td>
</tr>

<tr>
<td>Amount</td>
<td>{{ qr_result.upi.amount }}</td>
</tr>

</table>


<h3>🔎 Payment Security Checks</h3>

{% if qr_result.upi.reasons %}

{% for reason in qr_result.upi.reasons %}

<div class="signal">
• {{ reason }}
</div>

{% endfor %}

{% else %}

<div class="success">
✅ UPI payment payload structure detected successfully.
</div>

{% endif %}


<div class="warning">

<strong>Recommended Action</strong>

<br><br>

{{ qr_result.upi.action }}

</div>


{% elif qr_result.type == "EMAIL" %}

<div class="info">
📧 This QR contains an email destination.
Verify the recipient before sending sensitive information.
</div>


{% elif qr_result.type == "PHONE" %}

<div class="info">
📞 This QR contains a phone number.
Verify the number before calling.
</div>


{% elif qr_result.type == "WI-FI" %}

<div class="info">
📶 This QR contains Wi-Fi configuration information.
Only scan Wi-Fi QRs from trusted sources.
</div>


{% elif qr_result.type == "CONTACT" %}

<div class="info">
👤 This QR contains contact information.
Verify the source before saving the contact.
</div>


{% else %}

<div class="info">
📝 This QR contains text or another payload type.
QRShield decoded it successfully but it is not a web destination.
</div>

{% endif %}

</div>

{% endif %}


<div class="footer">

🛡️ QRShield • AI × Cybersecurity Mini Hackathon 2026

<br>

<span class="small">
Prototype using synthetic demonstration data.
Results are advisory and not a guarantee of safety.
</span>

</div>

</div>

</body>

</html>
"""


# =========================================================
# MAIN ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    url_result = None
    qr_result = None
    entered_url = ""

    if request.method == "POST":

        action = request.form.get(
            "action",
            ""
        )

        # -------------------------------------------------
        # URL ANALYSIS
        # -------------------------------------------------

        if action == "url":

            entered_url = request.form.get(
                "url",
                ""
            ).strip()

            if entered_url:

                url_result = analyze_url(
                    entered_url
                )

        # -------------------------------------------------
        # QR ANALYSIS
        # -------------------------------------------------

        elif action == "qr":

            uploaded_file = request.files.get(
                "qr_file"
            )

            if uploaded_file:

                file_bytes = uploaded_file.read()

                decoded = decode_qr(
                    file_bytes
                )

                if decoded:

                    qr_type = detect_qr_type(
                        decoded
                    )

                    qr_result = {
                        "content": decoded,
                        "type": qr_type,
                        "url_result": None,
                        "upi": None
                    }

                    if qr_type == "WEB URL":

                        qr_result["url_result"] = (
                            analyze_url(decoded)
                        )

                    elif qr_type == "UPI PAYMENT":

                        qr_result["upi"] = (
                            analyze_upi(decoded)
                        )

                else:

                    qr_result = {
                        "content": "",
                        "type": "UNKNOWN",
                        "url_result": None,
                        "upi": None
                    }

    return render_template_string(
        HTML,
        url_result=url_result,
        qr_result=qr_result,
        entered_url=entered_url
    )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
