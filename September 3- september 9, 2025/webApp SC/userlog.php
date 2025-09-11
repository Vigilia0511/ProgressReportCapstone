<?php
session_start();

$latestUser = isset($_SESSION['registered_user']) ? $_SESSION['registered_user'] : "No user registered";
?>



<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>User Login - Biometric</title>
<style>
/* (Your CSS unchanged) */
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
}
.container {
    background: white;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    max-width: 480px;
    width: 100%;
    padding: 32px;
    text-align: center;
}
.logo-container {
    background-color: #f1f5f9;
    border-radius: 60px;
    padding: 20px;
    margin: 0 auto 24px;
    width: 100px;
    display: flex;
    justify-content: center;
    align-items: center;
}
.logo-container img {
    height: 80px;
    width: 80px;
}
.user-info {
    background: #f7fafc;
    padding: 16px;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    justify-content: center;
    align-items: center;
}
.user-icon {
    background: #667eea;
    color: white;
    font-size: 16px;
    padding: 6px;
    width: 32px;
    height: 32px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.user-details label {
    display: block;
}
.user-details .name {
    font-size: 14px;
    font-weight: bold;
    color: #2d3748;
}
.user-details .status {
    font-size: 13px;
    color: #718096;
}
#error-message {
    color: red;
    margin-bottom: 16px;
}
#authButton {
    background: #667eea;
    color: white;
    font-weight: bold;
    padding: 12px;
    border: none;
    border-radius: 12px;
    width: 100%;
    cursor: pointer;
    font-size: 16px;
    margin-bottom: 24px;
}
.register-link {
    font-size: 15px;
}
.register-link a {
    color: #667eea;
    font-weight: bold;
    text-decoration: none;
}
#registrationForm {
    display: none;
}
</style>
</head>
<body>

<div class="container">
    <div class="logo-container">
        <img src="image/logo.PNG" alt="Logo" />
    </div>

    <h1>Welcome Back</h1>
    <p>Sign in with your biometric</p>

    <div id="error-message"></div>

    <div class="user-info">
        <div class="user-icon">👤</div>
        <div class="user-details">
            <label class="name">Current User</label>
            <label class="status"><?= htmlspecialchars($latestUser) ?></label>
        </div>
    </div>

    <button id="authButton">Register Biometric</button>

    <form id="registrationForm">
        <p>Biometric registered successfully! Redirecting...</p>
    </form>

    <div class="register-link">
        <span>Don't have an account? </span><a href="userreg.php">Register</a>
    </div>
</div>

<script>
async function runBiometricAuth() {
    if (!window.PublicKeyCredential) {
        alert("WebAuthn is not supported in this browser.");
        return;
    }

    const challenge = new Uint8Array(32);
    window.crypto.getRandomValues(challenge);

    try {
        const credential = await navigator.credentials.create({
            publicKey: {
                challenge: challenge,
                rp: { name: "Biometric Demo" },
                user: {
                    id: new Uint8Array(16), // Replace with real user ID bytes from server ideally
                    name: "<?= htmlspecialchars($latestUser) ?>", // Use current username
                    displayName: "Biometric User"
                },
                pubKeyCredParams: [{ type: "public-key", alg: -7 }],
                authenticatorSelection: {
                    authenticatorAttachment: "platform",
                    userVerification: "required"
                },
                timeout: 60000,
                attestation: "none"
            }
        });

        console.log("Biometric success:", credential);

        // Prepare data to send to server
        const credentialJson = {
            id: credential.id,
            rawId: arrayBufferToBase64(credential.rawId),
            type: credential.type,
            response: {
                clientDataJSON: arrayBufferToBase64(credential.response.clientDataJSON),
                attestationObject: arrayBufferToBase64(credential.response.attestationObject)
            }
        };

        // POST credential to backend for registration & approval check
        const resp = await fetch('registerBiometric.php', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                username: "<?= htmlspecialchars($latestUser) ?>",
                credential: credentialJson
            })
        });

        const data = await resp.json();

        if (data.isApproved) {
            document.getElementById("authButton").style.display = "none";
            document.getElementById("registrationForm").style.display = "block";
            // Redirect after short delay so user sees success message
            setTimeout(() => {
                window.location.href = "allnotif.php";
            }, 2000);
        } else {
            alert("Account not approved. Contact admin.");
        }

    } catch (err) {
        console.error("Authentication failed:", err);
        alert("Biometric authentication failed or was canceled.");
    }
}

function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let b of bytes) {
        binary += String.fromCharCode(b);
    }
    return window.btoa(binary);
}

document.getElementById("authButton").addEventListener("click", runBiometricAuth);
</script>

</body>
</html>
