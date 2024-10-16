import sys
import cv2
import asyncio
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel
from PySide6.QtCore import QTimer, Qt, Signal, QObject, QThread
from PySide6.QtGui import QImage, QPixmap

# Import your existing classes (adjust the import statements as needed)
from QuickAgent import LanguageModelProcessor, TextToSpeech, TranscriptCollector, get_transcript

class TranscriptionWorker(QObject):
    finished = Signal()
    transcription = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.running = True
        asyncio.run(self.transcribe())

    async def transcribe(self):
        transcript_collector = TranscriptCollector()

        def handle_full_sentence(full_sentence):
            self.transcription.emit(full_sentence)

        while self.running:
            await get_transcript(handle_full_sentence)

        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Interaction App")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Webcam display
        self.webcam_label = QLabel()
        self.layout.addWidget(self.webcam_label)

        # Text display area
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.layout.addWidget(self.text_display)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Interaction")
        self.stop_button = QPushButton("Stop Interaction")
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        self.layout.addLayout(button_layout)

        # Connect buttons to functions
        self.start_button.clicked.connect(self.start_interaction)
        self.stop_button.clicked.connect(self.stop_interaction)

        # Initialize webcam
        self.webcam = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

        # Initialize other components
        self.llm = LanguageModelProcessor()
        self.tts = TextToSpeech()

        self.interaction_active = False
        self.transcription_thread = None
        self.transcription_worker = None

    def update_frame(self):
        ret, frame = self.webcam.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.webcam_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))

    def start_interaction(self):
        if not self.interaction_active:
            self.interaction_active = True
            self.text_display.append("Starting interaction...")
            
            self.transcription_worker = TranscriptionWorker()
            self.transcription_thread = QThread()
            self.transcription_worker.moveToThread(self.transcription_thread)
            
            self.transcription_thread.started.connect(self.transcription_worker.run)
            self.transcription_worker.finished.connect(self.transcription_thread.quit)
            self.transcription_worker.transcription.connect(self.process_speech)
            
            self.transcription_thread.start()

    def stop_interaction(self):
        if self.interaction_active:
            self.interaction_active = False
            self.text_display.append("Stopping interaction...")
            
            if self.transcription_worker:
                self.transcription_worker.running = False

    def process_speech(self, text):
        self.text_display.append(f"You said: {text}")
        response = self.llm.process(text)
        self.text_display.append(f"AI response: {response}")
        self.tts.speak(response)

    def closeEvent(self, event):
        self.stop_interaction()
        self.webcam.release()
        if self.transcription_thread:
            self.transcription_thread.quit()
            self.transcription_thread.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())