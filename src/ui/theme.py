"""Hoja de estilos minimalista y accesible (baja carga cognitiva, alto contraste)."""

APP_STYLESHEET = """
QWidget {
    font-family: "Segoe UI", "Inter", "DejaVu Sans", Arial, sans-serif;
    color: #0F172A;
    font-size: 13px;
}

QMainWindow, QDialog {
    background-color: #F8FAFC;
}

QFrame#card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

QFrame#topBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
}

QFrame#bottomBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E2E8F0;
}

QFrame#floatingToolbar {
    background-color: #FFFFFF;
    border: 1.5px solid #CBD5E1;
    border-radius: 10px;
}

QLabel#title {
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
}

QLabel#subtitle {
    font-size: 12px;
    color: #475569;
}

QLabel#sectionHeader {
    font-size: 13px;
    font-weight: 700;
    color: #1E293B;
}

QLabel#taskInstruction {
    font-size: 13px;
    font-weight: 600;
    color: #0F172A;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 10px;
    color: #0F172A;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1.5px solid #2563EB;
}

QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
    color: #1E293B;
}

QPushButton:hover {
    background-color: #F1F5F9;
    border-color: #94A3B8;
}

QPushButton:disabled {
    background-color: #F8FAFC;
    color: #94A3B8;
    border-color: #E2E8F0;
}

QPushButton#primaryButton {
    background-color: #1E293B;
    color: #FFFFFF;
    border: none;
    padding: 9px 18px;
}

QPushButton#primaryButton:hover {
    background-color: #0F172A;
}

QPushButton#primaryButton:disabled {
    background-color: #94A3B8;
}

QPushButton#accentButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    padding: 8px 16px;
}

QPushButton#accentButton:hover {
    background-color: #1D4ED8;
}

QPushButton#dangerButton {
    background-color: #FEF2F2;
    color: #B91C1C;
    border: 1px solid #FECACA;
}

QPushButton#dangerButton:hover {
    background-color: #FEE2E2;
}

QProgressBar {
    border: 1px solid #CBD5E1;
    border-radius: 5px;
    background-color: #F1F5F9;
    text-align: center;
    height: 16px;
    font-size: 11px;
}

QProgressBar::chunk {
    background-color: #16A34A;
    border-radius: 4px;
}
"""
