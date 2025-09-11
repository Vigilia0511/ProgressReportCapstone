<?php
if ($_SERVER["REQUEST_METHOD"] === "POST") {
  $name = $_POST["name"] ?? "";
  $status = $_POST["status"] ?? "";

  $mysqli = new mysqli("localhost", "root", "", "smartdb");
  if ($mysqli->connect_error) {
    die("Connection failed: " . $mysqli->connect_error);
  }

  $stmt = $mysqli->prepare("UPDATE users SET is_approved = ? WHERE username = ?");
  $stmt->bind_param("is", $status, $name);

  if ($stmt->execute()) {
    echo "Status updated successfully.";
  } else {
    echo "Error updating status.";
  }

  $stmt->close();
  $mysqli->close();
}
?>
