const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
    const win = new BrowserWindow({
        width: 1280,
        height: 800,
        minWidth: 1024,
        minHeight: 600,
        menuBarVisible: true,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
        },
    });

    // Ubuntu yhteensopiva maximizointi
    win.maximize();       // toimii Win + Linux + Mac
    win.show();           // pakottaa ikkunan näkyviin Ubuntussa

    // Vite dev-server
    win.loadURL('http://localhost:5173');
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    // Linux ja Windows: sulje app kun ikkunat suljetaan
    if (process.platform !== 'darwin') app.quit();
});
