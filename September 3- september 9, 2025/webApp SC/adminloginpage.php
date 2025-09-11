
<?php if (isset($_GET['error']) && $_GET['error'] == '1'): ?>
<script>
  window.onload = function() {
    alert('Invalid credentials. Please try again.');
  };
</script>
<?php endif; ?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Admin Portal</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="css/styleforadminlogin.css">
</head>
<body>
    <div class="gradient-bg">
        <div class="login-container">
            <!-- Logo -->
            <div class="logo-wrapper">
                <img src="image/logo.png" alt="Logo" class="logo">
            </div>

            <!-- Title Section -->
            <div class="title-section">
                <h2>Admin Portal</h2>
                <p>Sign in to your account</p>
            </div>

            <!-- Biometric Section -->
            <div class="biometric-section" id="biometricSection" style="display:none;">
                <button onclick="useBiometric()" class="biometric-button">🔐 Use Biometric Authentication</button>
                <div class="divider">
                    <hr>
                    <span>or continue with credentials</span>
                </div>
            </div>

            <!-- Login Form -->
            <form method="POST" action="checkadmin.php">
                <label for="username">Admin ID</label>
                <input type="text" id="username" name="username" placeholder="Enter your admin ID" required>

                <label for="password">Password</label>
                <input type="password" id="password" name="password" placeholder="Enter your password" required>

                <button type="submit" class="login-button">Sign In</button>
            </form>

            <!-- Loading Indicator -->
            <div class="loading-indicator" id="loadingIndicator" style="display:none;">
                <div class="spinner"></div>
            </div>

            <!-- Security Notice -->
            <div class="security-notice">
                <span>🔒</span>
                <small>This is a secure admin area. All login attempts are monitored.</small>
            </div>
        </div>
    </div>

    <script>
        function useBiometric() {
            alert('Biometric authentication not implemented in this demo.');
        }

        document.querySelector("form").addEventListener("submit", () => {
            document.getElementById("loadingIndicator").style.display = "block";
        });
    </script>
</body>
</html>
