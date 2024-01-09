import cv2
from ultralytics import YOLO
from imutils.video import VideoStream
import sqlite3
from sqlite3 import Error
import time
import serial

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn
def send_data_logging(conn, task):
    sql ='''INSERT INTO table_adjuster_valve_hla(name_line, name_part, quantity, delay, status, andon) VALUES(?,?,?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()
    return cur.lastrowid
ser_arduino = serial.Serial(
    port='/dev/ttyACM0',
    baudrate = 9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
model = YOLO('/home/otics/on/project_pt_otics_ai_hla/best.pt')
rtsp_url = "rtsp://admin:pt_otics1*@192.168.1.108"
vidio_streaming = VideoStream(rtsp_url).start()
database = r"/home/otics/on/project_pt_otics_ai_hla/database/database_project_camai.db"
def main():
    while True:
        conn = create_connection(database)
        frame = vidio_streaming.read()
        results = model(frame)
        annotated_frame = results[0].plot()
        result = results[0]
        sum_hla = 0
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            if class_id == "hla":
                sum_hla += 1
        if(sum_hla == 192):
            ser_arduino.write(b"*oke_andon,192,Unit-Full-#")
            task = ("Packing_HLA", "Adjuster Valve HLA", str(sum_hla), "-", "1", "oke_andon")
            send_data_logging(conn, task)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            frame = cv2.resize(frame, (750, 500))
            file_name = f"/home/otics/on/project_pt_otics_ai_hla/runs/hasil_oke/HLA_{timestamp}.png"        
            print("Gambar berhasil disimpan:", file_name)
            cv2.imwrite(file_name, frame)          
        cv2.imshow("Hasil Deteksi", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vidio_streaming.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
