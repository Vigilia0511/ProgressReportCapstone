<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SecureAccess</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.6s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            width: 90%;
            max-width: 400px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .logo {
            background: #f1f5f9;
            border-radius: 50%;
            width: 120px;
            height: 120px;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0 auto 20px;
        }

        .logo img {
            width: 80px;
            height: 80px;
            object-fit: contain;
        }

        h1 {
            font-size: 2rem;
            color: #1e293b;
            margin-bottom: 10px;
        }

        p.subtitle {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 30px;
            line-height: 1.4;
        }

        .button {
            display: block;
            width: 100%;
            padding: 16px;
            margin: 10px 0;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: bold;
            color: white;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .button.admin {
            background-color: #3b82f6;
        }

        .button.user {
            background-color: #10b981;
        }

        .button:hover {
            transform: scale(0.98);
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }

        .section-label {
            font-size: 1.1rem;
            font-weight: bold;
            color: #334155;
            margin-bottom: 20px;
        }

        hr {
            border: none;
            height: 1px;
            background: #e2e8f0;
            margin: 20px 0;
        }

        .footer {
            font-size: 0.75rem;
            color: #94a3b8;
        }

        @media (max-width: 480px) {
            .container {
                padding: 30px 20px;
            }

            .logo {
                width: 100px;
                height: 100px;
            }

            .logo img {
                width: 60px;
                height: 60px;
            }

            h1 {
                font-size: 1.8rem;
            }

            p.subtitle {
                font-size: 0.85rem;
            }

            .section-label {
                font-size: 1rem;
            }

            .button {
                font-size: 0.95rem;
                padding: 14px;
            }

            .footer {
                font-size: 0.7rem;
            }
        }
    </style>
</head>
<body>

    <div class="container">
        <div class="logo">
            <img src="image/logo.png" alt="Logo" onerror="this.onerror=null; this.src='fallback-logo.png';">
        </div>

        <h1>SecureAccess</h1>
        <p class="subtitle">Layered Security Meets Trusted Simplicity</p>

        <p class="section-label">Choose Your Access Level</p>

        <form method="POST" action="adminloginpage.php">
            <button type="submit" class="button admin">👤 Administrator Access</button>
        </form>

        <form method="POST" action="userreg.php">
            <button type="submit" class="button user">👥 Standard User Access</button>
        </form>

        <hr>

        <div class="footer">
            Secure • Reliable • User-Friendly
        </div>
    </div>

    <script>
        // Optional click animation
        document.querySelectorAll('.button').forEach(button => {
            button.addEventListener('mousedown', () => {
                button.style.transform = 'scale(0.95)';
            });
            button.addEventListener('mouseup', () => {
                button.style.transform = 'scale(1)';
            });
            button.addEventListener('mouseleave', () => {
                button.style.transform = 'scale(1)';
            });
        });
    </script>

</body>
</html>
