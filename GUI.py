
from tkinter import *
from tkinter import Tk,filedialog
import os
import numpy as np
from PIL import Image
import cv2


root=Tk() # membuat window
root.iconbitmap("icon/icon.ico") # merubah icon
root.geometry('326x65') # merubah ukuran window
root.title('Face Recognition') # Merubah judul

pesan = "selamat datang"
#####################    MEMBUAT FUNGSI TOMBOL KETIKA DI TEKAN     #####################
def buatFolder(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)



def detect():
    # Membuat Local Binary Patterns Histograms untuk face recognization
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    buatFolder("trainer/")

    # muat mode yang data yang sudah di simpan pada recognizer
    recognizer.read('trainer/trainer.yml')

    cascadePath = "face-detect.xml"

    # Create classifier from prebuilt model
    faceCascade = cv2.CascadeClassifier(cascadePath);

    # mengatur font style text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # inisiasi camera
    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # Loop
    while True:
        # membaca camera
        ret, im = cam.read()

        # merubah gambar menjadi grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # mendeteksi wajah pada gambar
        wajah = faceCascade.detectMultiScale(gray, 1.2, 5)

        # perulangan untun mencocokan wajah dengan data training
        for (x, y, w, h) in wajah:

            # menandai pada bagian wajah dengan kotak / persegi panjang
            cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

            # mengenali id wajah
            Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # mengecek apakan id wajah sudah ada
            if Id is 1:
                Id = "Ni'am {0:.2f}%".format(round(100 - confidence, 2))
            if Id is 2:
                Id = "Arya {0:.2f}%".format(round(100 - confidence, 2))
            if Id is 3:
                Id = "Lali {0:.2f}%".format(round(100 - confidence, 2))
            if Id is 4:
                Id = "Mawa {0:.2f}%".format(round(100 - confidence, 2))
            if Id is 5:
                Id = "Wiranto {0:.2f}%".format(round(100 - confidence, 2))
            if Id is 6:
                Id = "Farid {0:.2f}%".format(round(100 - confidence, 2))

            print(Id)
            # menampilkan teks pada wajah yang terdeteksi
            cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
            cv2.putText(im, str(Id), (x, y - 40), font, 1, (255, 255, 255), 3)

        # menampilkan video frame untuk user
        cv2.imshow('Pengujian Pengenalan Wajah', im)

        # tekan 'q' untuk menghentikan program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # keluar dari camera
    cam.release()

    # keluar dari semua jendela
    cv2.destroyAllWindows()
def new():

    # Start capturing video
    vid_cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # Pendeteksi object video stream using Haarcascade Frontal Face
    face_detector = cv2.CascadeClassifier('face-detect.xml')

    # memberikan id untuk masing2 wajah
    face_id = input("masukkan Id baru : ")

    # inisialisasi jumlah gambar
    jumlah = 0

    buatFolder("dataset/")

    # Membuat Perulangan untuk mengambil 100 gambar wajah
    while (True):

        # Mengambil video frame
        _, image_frame = vid_cam.read()

        # Merubah video frame menjadi gambar grayscale
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        # Mendeteksi wajah pada videocam
        wajah = face_detector.detectMultiScale(gray, 1.3, 5)

        # mengambil gambar yang sudah di crop dan menyimpannya pada forder dataset
        for (x, y, w, h) in wajah:
            # Meng-crop hanya pada bagian wajah dengan sebuah kotak biru dengan ketebalan 2
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # nemabahkan nilai count (jumlah gambar pada id tertentu)
            jumlah += 1

            # Menyimpan gambar yang telah di tangkap pada folder dataset dengan nama sesuai ID dan count
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(jumlah) + ".jpg", gray[y:y + h, x:x + w])

            # Menampilkan video camera untuk untuk user yang akan di ambil gambar wajahnya
            cv2.imshow('Pengambilan Data Wajah', image_frame)

        # tekan q untuk menghentikan video frame
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # jika gambar yang diambi sudah mencapai 100 maka video frame akan berhenti secara otomatis
        elif jumlah > 100:
            break

    # menghentikan kamera
    vid_cam.release()

    # keluar dari semua jendela program
    cv2.destroyAllWindows()
def training():
    print("Prosses training data baru...")
    # Create Local Binary Patterns Histograms for face recognization
    # Buat Histogram pola biner lokal untuk pengenalan wajah
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Using prebuilt frontal face training model, for face detection
    # Menggunakan model pelatihan wajah frontal prebuilt, untuk deteksi wajah
    detector = cv2.CascadeClassifier("face-detect.xml");

    # Create method to get the images and label data | buat metede untuk mendapatkan gambar dan label data
    def getImagesAndLabels(path):
        # Get all file path | dapatkan semua jalur file
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

        # Initialize empty face sample | inisialisasi sampel wajah kosong
        faceSamples = []

        # Initialize empty id | inisialisasi id kosong
        ids = []


        # Loop all the file path | perulangan semua jalur file
        for imagePath in imagePaths:

            # Get the image and convert it to grayscale | dapatkan gambar dan ubah menjadi skala abu-abu
            PIL_img = Image.open(imagePath).convert('L')

            # PIL image to numpy array | gambar PIL ke array numpy
            img_numpy = np.array(PIL_img, 'uint8')

            # Get the image id | dapatkan ID gambar
            id = int(os.path.split(imagePath)[-1].split(".")[1])

            # Get the face from the training images | dapatkan wajad dari gambar training
            faces = detector.detectMultiScale(img_numpy)

            # Loop for each face, append to their respective ID | loop setiap wajah, tambahkan id masing-masing
            for (x, y, w, h) in faces:
                # Add the image to face samples | tambahkan gambar ke sampel wajah
                faceSamples.append(img_numpy[y:y + h, x:x + w])

                # Add the ID to IDs | tambahkan ID ke ID
                ids.append(id)

        # Pass the face array and IDs array | lulus array wajah dan array ID
        return faceSamples, ids

    # dapatkan wajah dan ID
    faces, ids = getImagesAndLabels('dataset')

    # latih model menggunakan wajah dan ID
    recognizer.train(faces, np.array(ids))

    # simpan medel ke dalam trainer.yml
    buatFolder('trainer/')
    recognizer.save('trainer/trainer.yml')
    print("training telah selesai")
def openFile():
    root.filename = filedialog.askopenfilename(initialdir="dataset/", title="Select file",filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    print(root.filename)

# membuat status bar
status = Label(root, text = pesan, bd = 1, relief = SUNKEN, anchor = W)
status.pack(side = BOTTOM, fill = X)

#======================================================================================#
#                             MEMBUAT TOMBOL BERGAMBAR                                 #
#======================================================================================#

# tombol uji face recognition
img_detect = PhotoImage(file="img/face_detec.png").subsample(15, 15)
btn_detect = Button(root, image=img_detect, compound=LEFT, command=detect, text='Detect').pack(side=LEFT)

# tombol untuk menambahkan data
img_new = PhotoImage(file="img/add_data.png").subsample(15, 15)
btn_new = Button(root, image=img_new, compound=LEFT, command=new, text='New').pack(side=LEFT)

# tombol untuk includ data training
img_training = PhotoImage(file="img/training.png").subsample(15, 15)
btn_training = Button(root, image=img_training, compound=LEFT, command=training, text='Training').pack(side=LEFT)

# tombol untuk melihat data training
img_open_file = PhotoImage(file="img/files.png").subsample(15, 15)
btn_open_file = Button(root, image=img_open_file, compound=LEFT, command=openFile, text='Open file').pack(side=LEFT)



root.mainloop()