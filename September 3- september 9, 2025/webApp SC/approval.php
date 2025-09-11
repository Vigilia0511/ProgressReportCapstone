<?php
// Database connection
$mysqli = new mysqli("localhost", "root", "", "smartdb");
if ($mysqli->connect_error) {
  die("Connection failed: " . $mysqli->connect_error);
}

// Fetch users
$result = $mysqli->query("SELECT *, is_approved FROM users ORDER BY timestamp DESC");
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>User Approval Panel</title>
<style>
  * {
    box-sizing: border-box;
  }

  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background: #f4f6f8;
  }

  .header {
    background: linear-gradient(to right, #2980b9, #6dd5fa);
    color: white;
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .logo {
    display: flex;
    align-items: center;
  }

  .logo img {
    width: 40px;
    margin-right: 10px;
  }

  .menu-icon {
    width: 30px;
    cursor: pointer;
  }

  .icon-button {
    background: none;
    border: none;
    margin-left: 10px;
    cursor: pointer;
  }

  .icon-button img {
    width: 24px;
    height: 24px;
  }

  .overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 999;
    display: none;
  }

  .sidebar {
    position: fixed;
    top: 0;
    right: 0;
    width: 250px;
    height: 100vh;
    background-color: #2A6FA4;
    padding: 20px;
    display: flex;
    flex-direction: column;
    transform: translateX(250px);
    transition: transform 0.3s ease;
    z-index: 1000;
    color: white;
  }

  .sidebar.visible {
    transform: translateX(0);
  }

  .sidebar-header {
    text-align: center;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.3);
  }

  .sidebar-header .user-icon {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    padding: 10px;
    margin-bottom: 10px;
  }

  .btn-link {
    display: flex;
    align-items: center;
    padding: 8px 16px;
    background-color: transparent;
    color: white;
    text-decoration: none;
    border-radius: 8px;
    font-size: 14px;
    border: none;
    transition: background 0.2s ease-in-out;
  }

  .btn-link:hover {
    background-color: #0056b3;
  }

  .sidebar-icon {
    width: 20px;
    height: 20px;
  }

  .dropdown-arrow {
    margin-left: auto;
    transition: transform 0.3s ease;
    width: 16px;
    height: 16px;
    fill: white;
  }

  .dropdown-arrow.open {
    transform: rotate(90deg);
  }

  .settings-dropdown {
    overflow: hidden;
    max-height: 0;
    transition: max-height 0.3s ease;
    background-color: transparent;
    border-radius: 8px;
    margin-top: 5px;
    padding: 0 15px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .settings-dropdown.open {
    max-height: 200px;
    padding-top: 15px;
    padding-bottom: 15px;
  }

  .settings-dropdown .toggle-label {
    display: flex;
    justify-content: space-between;
    color: white;
  }

  .sidebar-toggle-group {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 15px;
    margin: 20px 0;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .panel {
    background-color: white;
    margin: 20px auto;
    padding: 20px;
    max-width: 800px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  h2 {
    text-align: center;
    margin-bottom: 20px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  thead {
    background-color: #f0f0f0;
  }

  th, td {
    padding: 14px 12px;
    text-align: left;
    border-bottom: 1px solid #e0e0e0;
  }

  th {
    font-weight: 600;
  }

  .status-approved {
    color: green;
    font-weight: bold;
  }

  .status-denied {
    color: red;
    font-weight: bold;
  }

  .btn {
    padding: 8px 16px;
    border: none;
    border-radius: 20px;
    color: white;
    cursor: pointer;
    font-size: 14px;
  }

  .approve-btn {
    background-color: green;
  }

  .disallow-btn {
    background-color: red;
  }

  button {
    background-color: transparent;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s;
    display: block;
    text-align: left;
    padding: 10px;
    color: white;
  }

  button:hover {
    background-color: rgba(0, 123, 255, 0.2);
  }

  .controls {
    background: linear-gradient(to right, #6dd5fa, #2980b9);
    padding: 10px 15px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #333;
  }

  .toggle-switch {
    position: relative;
    display: inline-block;
    width: 50px;
    height: 24px;
  }

  .toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .toggle-switch {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 24px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-switch .slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: 0.4s;
  border-radius: 24px;
}

.toggle-switch .slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: 0.4s;
  border-radius: 50%;
}

.toggle-switch input:checked + .slider {
  background-color: #4caf50;
}

.toggle-switch input:checked + .slider:before {
  transform: translateX(26px);
}

body.dark {
  background-color: #121212;
  color: #f0f0f0;
}

body.dark .panel {
  background-color: #1e1e1e;
  color: #f0f0f0;
}

body.dark .header {
  background: linear-gradient(to right, #222, #555);
}

body.dark table {
  background-color: #2c2c2c;
}

body.dark th,
body.dark td {
  border-color: #444;
}

body.dark .btn-link:hover {
  background-color: #333;
}


  /* Responsive Mobile Layout */
  @media (max-width: 600px) {
    table, thead, tbody, th, td, tr {
      display: block;
      width: 100%;
    }

    thead {
      display: none;
    }

    tr {
      background-color: #fff;
      margin-bottom: 15px;
      border-radius: 8px;
      padding: 10px;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }

    td {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px;
      border-bottom: 1px solid #eaeaea;
    }

    td::before {
      content: attr(data-label);
      font-weight: 600;
      flex-basis: 45%;
      color: #333;
    }

    td:last-child {
      border-bottom: none;
    }
  }
</style>

</head>
<body>
  <div class="header">
    <div class="logo">
      <img src="image/logo.png" alt="Logo" />
    </div>
    <img src="image/menu.png" class="menu-icon" alt="Menu" id="menuIcon" />
  </div>

  <div class="panel">
    <h2>User Approval Panel</h2>
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Status</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        <?php while($row = $result->fetch_assoc()): ?>
        <tr>
          <td data-label="Name"><?= htmlspecialchars($row['username']) ?></td>
          <td data-label="Status">
            <span class="status <?= $row['is_approved'] ? 'status-approved' : 'status-denied' ?>">
              <?= $row['is_approved'] ? '✔ Approved' : '✖ Not Approved' ?>
            </span>
          </td>
          <td data-label="Action">
            <button 
              class="btn <?= $row['is_approved'] ? 'disallow-btn' : 'approve-btn' ?>" 
              onclick="toggleStatus(this)">
              <?= $row['is_approved'] ? 'Disallow' : 'Approve' ?>
            </button>
          </td>
        </tr>
        <?php endwhile; ?>
      </tbody>
    </table>
  </div>

  <div class="overlay" id="overlay"></div>

  <nav class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <img src="image/profile.png" alt="User Icon" class="user-icon" />
      <p>Welcome, <strong>ADMIN</strong></p>
    </div>
    <a href="allnotif.php" class="btn-link">
      <img src="image/home.png" class="sidebar-icon" /> Home
    </a>
    <a href="approval.php" class="btn-link">
      <img src="image/approval.png" class="sidebar-icon" /> Approval
    </a>
    <a href="images.php" class="btn-link">
      <img src="image/picture.png" class="sidebar-icon" /> Image
    </a>
    <button id="settingsToggle" aria-expanded="false" aria-controls="settingsDropdown">
      <img src="image/settings.png" class="sidebar-icon" />
      Settings
      <svg class="dropdown-arrow" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z"/>
      </svg>
    </button>
    <div class="settings-dropdown" id="settingsDropdown">
      <div class="toggle-label">
        <label for="darkModeToggle">Dark Mode</label>
        <label class="toggle-switch">
          <input type="checkbox" id="darkModeToggle" />
          <span class="slider"></span>
        </label>
      </div>
      <div class="toggle-label">
        <label for="biometricToggle">Enable Biometric</label>
        <label class="toggle-switch">
          <input type="checkbox" id="biometricToggle" checked />
          <span class="slider"></span>
        </label>
      </div>
    </div>
    <button id="logoutBtn">
      <img src="image/logout.png" class="sidebar-icon" /> Logout
    </button>
  </nav>

  <script>
document.getElementById('logoutBtn').addEventListener('click', function () {
  const confirmed = confirm("Are you sure you want to log out?");
  if (confirmed) {
    // Redirect to index.php after logging out
    window.location.href = "logout.php";
  }
});


    const menuIcon = document.getElementById('menuIcon');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('overlay');
    const darkModeToggle = document.getElementById('darkModeToggle');
    const settingsToggle = document.getElementById('settingsToggle');
    const settingsDropdown = document.getElementById('settingsDropdown');
    const dropdownArrow = settingsToggle.querySelector('.dropdown-arrow');

    menuIcon.addEventListener('click', () => {
      sidebar.classList.add('visible');
      overlay.style.display = 'block';
    });

    overlay.addEventListener('click', () => {
      sidebar.classList.remove('visible');
      overlay.style.display = 'none';
    });

    darkModeToggle.addEventListener('change', (e) => {
      document.body.classList.toggle('dark', e.target.checked);
    });

    settingsToggle.addEventListener('click', () => {
      const isOpen = settingsDropdown.classList.toggle('open');
      dropdownArrow.classList.toggle('open', isOpen);
      settingsToggle.setAttribute('aria-expanded', isOpen);
    });

    function toggleStatus(button) {
      const row = button.closest('tr');
      const name = row.querySelector('td[data-label="Name"]').textContent.trim();
      const statusCell = row.querySelector('.status');
      let newStatus = 0;

      if (button.textContent === 'Approve') {
        statusCell.textContent = '✔ Approved';
        statusCell.classList.remove('status-denied');
        statusCell.classList.add('status-approved');
        button.textContent = 'Disallow';
        button.classList.remove('approve-btn');
        button.classList.add('disallow-btn');
        newStatus = 1;
      } else {
        statusCell.textContent = '✖ Not Approved';
        statusCell.classList.remove('status-approved');
        statusCell.classList.add('status-denied');
        button.textContent = 'Approve';
        button.classList.remove('disallow-btn');
        button.classList.add('approve-btn');
        newStatus = 0;
      }

      fetch('update_status.php', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `name=${encodeURIComponent(name)}&status=${newStatus}`
      })
      .then(response => response.text())
      .then(data => {
        console.log('Server response:', data);
      })
      .catch(error => {
        console.error('Error:', error);
      });
    }
  </script>

</body>
</html>
