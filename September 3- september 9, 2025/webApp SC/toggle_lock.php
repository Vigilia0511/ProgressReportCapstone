<?php
header('Content-Type: application/json');

$action = $_GET['action'] ?? '';

$gpioPin = 6;  // Change to your actual relay control pin

// Export and set pin mode if needed (once)
// shell_exec("gpio -g mode $gpioPin out");

if ($action === 'unlock') {
    // Relay ON (depends on wiring: LOW might trigger it)
    shell_exec("gpio -g write $gpioPin 1");
    echo json_encode(['success' => true]);
} elseif ($action === 'lock') {
    // Relay OFF
    shell_exec("gpio -g write $gpioPin 0");
    echo json_encode(['success' => true]);
} else {
    echo json_encode(['success' => false, 'message' => 'Invalid action']);
}
