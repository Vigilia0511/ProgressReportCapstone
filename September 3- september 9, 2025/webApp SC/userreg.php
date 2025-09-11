<?php
session_start();

$error = $success = "";
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $fullname = trim($_POST['fullname']);

    if (strlen($fullname) < 2) {
        $error = "Please enter a valid full name (at least 2 characters).";
    } else {
        $servername = "localhost";
        $username = "root";
        $password = "";
        $dbname = "smartdb";

        $conn = new mysqli($servername, $username, $password, $dbname);

        if ($conn->connect_error) {
            $error = "Database connection failed: " . $conn->connect_error;
        } else {
            $stmt = $conn->prepare("SELECT COUNT(*) FROM users WHERE LOWER(username) = LOWER(?)");
            $stmt->bind_param("s", $fullname);
            $stmt->execute();
            $stmt->bind_result($count);
            $stmt->fetch();
            $stmt->close();

            if ($count > 0) {
                $error = "This name is already registered. Please use another or contact support.";
            } else {
                $stmt = $conn->prepare("INSERT INTO users (username, is_approved) VALUES (?, 0)");
                $stmt->bind_param("s", $fullname);
                if ($stmt->execute()) {
                    $_SESSION['registered_user'] = $fullname;  // Added this to save the user in session
                    header("Location: userlog.php");           // Redirect to userlog.php after registration
                    exit();
                } else {
                    $error = "Failed to register: " . $stmt->error;
                }
                $stmt->close();
            }
            $conn->close();
        }
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Biometric Registration</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Base Reset */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body, html {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            height: 100%;
        }

        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 16px;
        }

        .card {
            background: #fff;
            border-radius: 20px;
            padding: 32px 24px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 420px;
            text-align: center;
        }

        .logo-container img {
            width: 100px;
            height: 100px;
            background-color: #f1f5f9;
            border-radius: 50%;
            padding: 20px;
            margin-bottom: 20px;
        }

        h1 {
            font-size: 30px;
            color: #1a202c;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 18px;
            color: #718096;
            margin-bottom: 24px;
        }

        .info-section {
            display: flex;
            flex-direction: column;
            gap: 16px;
            margin-bottom: 24px;
        }

        .info-item {
            display: flex;
            align-items: center;
            gap: 12px;
            background-color: #f7fafc;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 16px;
            color: #4a5568;
        }

        .message {
            margin-bottom: 16px;
            padding: 12px;
            border-radius: 10px;
            font-size: 15px;
        }

        .message.error {
            background-color: #fed7d7;
            color: #c53030;
        }

        .message.success {
            background-color: #c6f6d5;
            color: #2f855a;
        }

        form input {
            width: 100%;
            padding: 14px;
            margin-bottom: 18px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            font-size: 16px;
        }

        form button, #authButton {
            width: 100%;
            padding: 14px;
            background-color: #667eea;
            color: white;
            font-size: 17px;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.1s ease-in-out, background-color 0.2s;
        }

        form button:hover, #authButton:hover {
            transform: scale(1.03);
            background-color: #5a67d8;
        }

        .divider {
            height: 1px;
            background-color: #e2e8f0;
            margin: 28px 0;
        }

        .login-link {
            font-size: 16px;
            color: #718096;
        }

        .login-link a {
            color: #667eea;
            font-weight: bold;
            text-decoration: none;
        }

        .login-link a:hover {
            text-decoration: underline;
        }

        @media (max-width: 480px) {
            .card {
                padding: 24px 18px;
            }

            h1 {
                font-size: 26px;
            }

            .subtitle {
                font-size: 16px;
            }

            form input, form button, #authButton {
                font-size: 16px;
            }

            .info-item {
                font-size: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="logo-container">
                <img src="image/logo.png" alt="Logo">
            </div>
            <h1>Create Account</h1>
            <p class="subtitle">Secure biometric registration</p>

            <div class="info-section">
                <div class="info-item">
                    <span class="icon">🛡️</span>
                    <span>Biometric authentication required</span>
                </div>
                <div class="info-item">
                    <span class="icon">⏳</span>
                    <span>Admin approval needed</span>
                </div>
            </div>

            <!-- Display PHP messages -->
            <?php if (!empty($error)): ?>
                <div class="message error"><?= htmlspecialchars($error) ?></div>
            <?php elseif (!empty($success)): ?>
                <div class="message success"><?= htmlspecialchars($success) ?></div>
            <?php endif; ?>

            <!-- Biometric auth button -->
            <button id="authButton">Authenticate with Biometrics</button>

            <!-- Form hidden until biometric is successful -->
            <form id="registrationForm" method="POST" action="userreg.php" style="display: none;">
                <input type="text" name="fullname" placeholder="Enter your full name..." required>
                <button type="submit">Register</button>
            </form>

            <div class="divider"></div>
            <p class="login-link">
                Already have an account? <a href="userlog.php">Sign In</a>
            </p>
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
                        id: new Uint8Array(16),
                        name: "biouser",
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
            document.getElementById("authButton").style.display = "none";
            document.getElementById("registrationForm").style.display = "block";

        } catch (err) {
            console.error("Authentication failed:", err);
            alert("Biometric authentication failed or was canceled.");
        }
    }

    document.getElementById("authButton").addEventListener("click", runBiometricAuth);
    </script>
</body>
</html>
