<?php
// registerBiometric.php
header('Content-Type: application/json');
$data = json_decode(file_get_contents('php://input'), true);

$username = $data['username'] ?? '';
$credential = $data['credential'] ?? null;

if (!$username || !$credential) {
    echo json_encode(['isApproved' => false, 'error' => 'Missing data']);
    exit;
}

// TODO: Save $credential data linked to $username in your DB here
// For now, just simulate success:

// Connect DB
$conn = new mysqli("localhost", "root", "", "smartdb");
if ($conn->connect_error) {
    echo json_encode(['isApproved' => false, 'error' => 'DB connection failed']);
    exit;
}

// Check if user is approved
$stmt = $conn->prepare("SELECT is_approved FROM users WHERE username = ?");
$stmt->bind_param("s", $username);
$stmt->execute();
$stmt->bind_result($isApproved);
if ($stmt->fetch() && $isApproved) {
    // TODO: Save credential to DB for this user here
    
    echo json_encode(['isApproved' => true]);
} else {
    echo json_encode(['isApproved' => false]);
}
$stmt->close();
$conn->close();
