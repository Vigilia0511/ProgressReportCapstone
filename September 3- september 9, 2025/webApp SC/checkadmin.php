<?php
// Example credentials
$valid_username = "051123";
$valid_password = "admin11";

// Handle login
if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $username = $_POST['username'] ?? '';
    $password = $_POST['password'] ?? '';

    if ($username === $valid_username && $password === $valid_password) {
        echo "<script>window.location.href='approval.php';</script>";
    } else {
header("Location: adminloginpage.php?error=1");
exit;

    }
}
?>

