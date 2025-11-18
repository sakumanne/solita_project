const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
    const win = new BrowserWindow ({
        width: 1000,
        height: 700,
        webPreferences: {
            nodeIntegration: false, //Turvasyitä pois
            contextIsolation: true,
        },
    });

    win.loadURL('http://localhost:5173'); // Vite dev-server
}

app.whenReady().then(createWindow);