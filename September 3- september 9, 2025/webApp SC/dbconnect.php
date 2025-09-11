<?php
$conn = new mysqli("localhost", "root", "", "smartdb");

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
?>
