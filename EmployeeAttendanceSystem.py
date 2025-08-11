# Importing required libraries
import cv2
import pandas as pd
import os
import numpy as np
import dlib
import shutil
import face_recognition
from math import hypot
from datetime import timedelta, datetime
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QTableWidgetItem
from PyQt5.uic import loadUi
from cryptography.fernet import Fernet
import xlsxwriter

# Class for the Login Module


class Login(QDialog):

    # Self function for opening Login Page GUI

    def __init__(self):
        super(Login, self).__init__()
        loadUi("GUI/login.ui", self)
        self.loginbutton.clicked.connect(self.loginfunction)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.createaccbutton.clicked.connect(self.gotocreate)
        self.invalid.setVisible(False)

    # Function to check authorization of the admin
    # Fernet is used for encryption and decryption of the data
    # Takes user id and password from user and checks for availability in data
    # User ids and passwords are present at the docs in file system by encrypting it

    def loginfunction(self):
        userid = self.userid.text()
        password = self.password.text()
        with open('docs/mykey.key', 'rb') as dec:
            key = dec.read()
        fer = Fernet(key)
        with open("docs/users.txt", 'rb') as ud:
            encdata = ud.read()
        endata = fer.decrypt(encdata)
        with open('docs/users.txt', 'wb') as wdec:
            wdec.write(endata)
        with open("docs/users.txt", 'r') as file:
            list_of_users = []
            while True:
                line = file.readline()
                if not line:
                    break
                list_of_users.append(line.split())
        with open('docs/users.txt', 'wb') as wdec:
            wdec.write(encdata)

        # if user present in our date
        if [userid, password] in list_of_users:
            homepage = HomePage()
            widget.addWidget(homepage)
            widget.setCurrentIndex(widget.currentIndex()+1)
        else:
            self.invalid.setVisible(True)
            self.userid.setText("")
            self.password.setText("")

    # function to go for Create account function

    def gotocreate(self):
        createacc = CreateAcc()
        widget.addWidget(createacc)
        widget.setCurrentIndex(widget.currentIndex()+1)

# Class for the Home Module


class HomePage(QDialog):

    # Self function for opening HomePage GUI

    def __init__(self):
        super(HomePage, self).__init__()
        loadUi("GUI/homepage.ui", self)
        self.take_attendence.clicked.connect(self.takeattendence)
        self.show_attendence.clicked.connect(self.showattendence)
        self.add_student.clicked.connect(self.addstudent)
        self.delete_student.clicked.connect(self.deletestudent)
        self.logout.clicked.connect(self.loginscreen)

    # function to go home page module

    def home_page(self):
        homepage = HomePage()
        widget.addWidget(homepage)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    # function to go take attendance module

    def takeattendence(self):
        take_attendence = TakeAttendence()
        widget.addWidget(take_attendence)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    # function to go show attendance/report module

    def showattendence(self):
        show_attendence = ShowAttendance()
        widget.addWidget(show_attendence)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    # function to go add student module

    def addstudent(self):
        addstu = AddEmployee()
        widget.addWidget(addstu)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    # function to go delete student module

    def deletestudent(self):
        delstu = DeleteEmployee()
        widget.addWidget(delstu)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    # function to go login module
    # logout function

    def loginscreen(self):
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)

# Class for Take Attendance Module


class TakeAttendence(QDialog):

    # Self function for opening Show take attendance GUI

    def __init__(self):
        self.myList = []
        self.classNames = []
        self.report = {}
        self.encodeListKnown = self.read_encoded_data()
        super(TakeAttendence, self).__init__()
        self.flag = True
        loadUi("GUI/take_attendence.ui", self)
        self.closecamera.setVisible(False)
        self.showcamera.setVisible(False)
        self.emp.setVisible(False)
        self.emp_2.setVisible(False)
        self.back.clicked.connect(HomePage.home_page)
        self.start.clicked.connect(self.takeattendence)

    # function to read the existing face datas of the employees

    def read_encoded_data(self):
        encoded_data = []
        with open('docs/encoded_data.txt') as f:
            for s in f.readlines():
                s1 = s.split("', [")
                temp_face = np.array([float(x.strip()) for x in s1[1].split(',')])
                encoded_data.append(temp_face.astype(np.float64))
                self.myList.append(s1[0])
                temp = s1[0].split('.')[0]
                self.classNames.append(temp)
                self.report[temp] = 'Absent'

        return encoded_data

    # function to start taking the attendance
    def takeattendence(self):
        morning_file = datetime.now().strftime('%d %B %Y')
        ss = pd.read_excel('Attendance/report.xlsx')
        d = ss.columns.to_list()
        if morning_file in d:
            self.emp_2.setVisible(True)
        else:
            self.emp_2.setVisible(False)
            self.start.setVisible(False)
            self.back.setVisible(False)
            self.showcamera.setVisible(True)
            self.visible = False
            self.stopattendence = False

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")

            open_count = [0] * len(self.myList)
            closed_count = [0] * len(self.myList)

            # markig attendance when student is present

            def mark_attendance(i_empid):
                self.report[i_empid] = 'Present'
                self.emp.setVisible(True)
                self.emp.setText(i_empid+" your attendance has been taken")

            # finding mid point of the eye.

            def midpoint(p1, p2):
                return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

            # Finding the blinking ratio of the eyes

            def get_blinking_ratio(eye_points, facial_landmarks):
                left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
                right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
                center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
                center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

                hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
                ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

                return hor_line_length / ver_line_length
            # opening camera
            # print("Hi")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            # print("Hi 2")
            # loop while the camera is running detects every single frame

            while self.flag:
                # capturing images from the camera and realizing them

                success, self.img = cap.read()
                img_s = cv2.resize(self.img, (0, 0), None, 0.25, 0.25)
                img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                faces_cur_frame = face_recognition.face_locations(img_s)
                encodes_cur_frame = face_recognition.face_encodings(img_s, faces_cur_frame)

                # algorithm for checking every single frame for faces

                for encodeFace, faceLoc in zip(encodes_cur_frame, faces_cur_frame):
                    matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace, 0.4)
                    face_dis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                    match_index = int(np.argmin(face_dis))
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(self.img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    self.emp.setVisible(False)

                    # Checking the captured face is there in the list or not

                    if matches[match_index]:
                        empid = self.classNames[match_index].upper()
                        cv2.putText(self.img, empid, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                        # for matched faces checking for the eyes to check their ratio

                        for face in faces:
                            landmarks = predictor(gray, face)
                            # finding each eye ratio
                            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
                            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
                            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

                            # if eye is open or not
                            if blinking_ratio < 4:
                                open_count[match_index] = open_count[match_index] + 1

                            # if eye closes
                            if blinking_ratio > 5.7:
                                closed_count[match_index] = closed_count[match_index] + 1

                            # when two blinks are done
                            if (closed_count[match_index] > 0) and (open_count[match_index] > 0):
                                mark_attendance(empid)
                    # if face is not there in our list

                    else:
                        cv2.putText(self.img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                                    2)

                self.showcamera.clicked.connect(self.show_camera)
                self.closecamera.clicked.connect(self.close_camera)
                self.stop.clicked.connect(self.stop_attendance)
                if self.visible:
                    cv2.imshow('b', self.img)
                    cv2.moveWindow('b', 600, 10)
                self.key = cv2.waitKey(1)

                # when stop attendance is clicked

                if self.key == 27 or self.stopattendence:
                    wb = xlsxwriter.Workbook('Attendance/temp.xlsx')
                    ws = wb.add_worksheet("temp")
                    date = datetime.now().strftime('%d %B %Y')
                    ws.write(0, 0, "Employee ID")
                    ws.write(0, 1, date)
                    index = 1
                    for entry in self.report.keys():
                        ws.write(index, 0, entry)
                        ws.write(index, 1, self.report[entry])
                        index += 1
                    wb.close()
                    initial_wb = 'Attendance/report.xlsx'
                    second_wb = 'Attendance/temp.xlsx'

                    df_initial = pd.read_excel(initial_wb)
                    df_second = pd.read_excel(second_wb)

                    df_3 = pd.merge(df_initial, df_second[['Employee ID', date]], on='Employee ID', how='left')
                    df_3.to_excel(initial_wb, index=False)
                    homepage = HomePage()
                    widget.addWidget(homepage)
                    widget.setCurrentIndex(widget.currentIndex() + 1)
                    break

            try:
                cv2.destroyWindow('b')
            except:
                pass
            cap.release()
            pass

    # fuction to stop attendance
    
    def stop_attendance(self):
        self.stopattendence = True
        self.emp.setVisible(False)

    # function to open camera

    def show_camera(self):
        self.visible = True
        self.closecamera.setVisible(True)
        self.showcamera.setVisible(False)

    # function to close camera

    def close_camera(self):
        try:
            cv2.destroyWindow('b')
        except:
            pass
        self.visible = False
        self.showcamera.setVisible(True)
        self.closecamera.setVisible(False)

# Class for Report Module


class ShowAttendance(QDialog):

    # Self function for opening Report GUI

    def __init__(self):
        super(ShowAttendance, self).__init__()
        loadUi("GUI/show_attendence.ui", self)
        self.back_btn.clicked.connect(HomePage.home_page)
        self.table.setVisible(False)
        self.invalid.setVisible(False)
        self.downloadbtn.setVisible(False)
        self.showbtn.clicked.connect(self.showattendence)

    # Function to perform show report operation

    def showattendence(self):
        d_num = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        if str(self.enddate.date().day()) in d_num:
            self.end_day = '0' + str(self.enddate.date().day())
        else:
            self.end_day = str(self.enddate.date().day())
        if str(self.startdate.date().day()) in d_num:
            self.start_day = '0' + str(self.startdate.date().day())
        else:
            self.start_day = str(self.startdate.date().day())
        m_name = ['', ' January ', ' February ', ' March ', ' April ', ' May ', ' June ', ' July ', ' August ',
                  ' September ', ' October ', ' November ', ' December ']
        sdate = self.start_day + m_name[self.startdate.date().month()] + str(self.startdate.date().year())
        edate = self.end_day + m_name[self.enddate.date().month()] + str(self.enddate.date().year())
        df = pd.read_excel('Attendance/report.xlsx')
        wb = df.columns.to_list()
        sd = pd.to_datetime(wb[1])
        ed = pd.to_datetime(wb[-1])
        # checking if entered dates present in the list
        if (self.startdate.date() <= self.enddate.date()):
            if(self.startdate.date() >= sd) and (ed >= self.enddate.date()):
                if not ((sdate in wb) and (edate in wb)):
                    while sdate not in wb:
                        tempdate = pd.to_datetime(sdate)
                        tempdate += timedelta(days=1)
                        sdate = tempdate.strftime('%d %B %Y')
                    while edate not in wb:
                        tempdate = pd.to_datetime(edate)
                        tempdate -= timedelta(days=1)
                        edate = tempdate.strftime('%d %B %Y')
                desired_headings = [wb[0]] + wb[wb.index(sdate): wb.index(edate) + 1]
                self.selected_columns = df.loc[:, desired_headings]
                self.selected_columns.to_excel('Attendance/attendance_report.xlsx')
                self.frame.setVisible(False)
                self.table.setVisible(True)
                self.downloadbtn.setVisible(True)
                self.downloadbtn.clicked.connect(self.downloadfun)
                self.selected_columns.fillna('NA', inplace=True)
                self.table.setRowCount(self.selected_columns.shape[0])
                self.table.setColumnCount(self.selected_columns.shape[1])
                self.table.setHorizontalHeaderLabels(self.selected_columns.columns)
                for row in self.selected_columns.iterrows():
                    values = row[1]
                    for col_index, value in enumerate(values):
                        table_item = QTableWidgetItem(value)
                        self.table.setItem(row[0], col_index, table_item)

            # if dates are not there in the data

            else:
                t = ("Enter Dates between " + wb[1] + "," + wb[-1])
                self.invalid.setText(t)
                self.invalid.setVisible(True)

        # if dates are not there in the data

        else:
            t = ("Enter Dates between " + wb[1] + "," + wb[-1])
            self.invalid.setText(t)
            self.invalid.setVisible(True)

    # Function to download the report

    def downloadfun(self):
        try:
            foldername = QFileDialog.getExistingDirectory(self, caption='select a folder', directory='C:')
            self.selected_columns.to_excel(foldername+'/attendance_report.xlsx', index=False)
        except:
            pass

# Class for Add Employee Module


class AddEmployee(QDialog):

    # Self function for opening Add Employee GUI

    def __init__(self):
        super(AddEmployee, self).__init__()
        loadUi("GUI/add_student.ui", self)
        self.frame2.setVisible(False)
        self.warning.setVisible(False)
        self.back_btn.clicked.connect(HomePage.home_page)
        self.add_image.clicked.connect(self.addimage)

    # function to go for the add image module

    def addimage(self):
        self.warning.setVisible(False)
        self.frame2.setVisible(True)
        self.success.setVisible(False)
        self.browse.clicked.connect(self.browsefiles)
        self.add.clicked.connect(self.takeimage)
        self.cancel.clicked.connect(self.cancel_add)

    # function to perform add image operation

    def takeimage(self):
        list1 = TakeAttendence()
        fname = self.filename.text()
        self.success.setVisible(False)
        # if employee is selected from files

        if not fname == '':
            # if employee face is detected

            try:

                # if employee not present in our data

                filename = fname.split("/")[-1]
                if filename not in list1.myList:
                    self.success.setVisible(True)
                    target = os.path.join(os.getcwd(), 'ImagesAttendance')
                    shutil.copy(self.fname[0], target)
                    # self.frame2.setVisible(False)
                    cur_img = cv2.imread(f'{fname}')
                    i_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
                    encode = list(face_recognition.face_encodings(i_img)[0])
                    with open('docs/encoded_data.txt', 'a') as f:
                        f.write("\n" + str([filename, encode])[2:-2])
                    with open('docs/encoded_data.txt', 'r') as f:
                        text = f.read().strip()
                        text = text.split('\n')
                        text.sort()
                        if(text[0] == ''):
                            text = text[1:]
                    with open('docs/encoded_data.txt', 'w') as f:
                        f.write('\n'.join(text))
                    self.filename.setText('')

                # if employee already present in our data

                else:
                    self.warning.setVisible(True)

            # if employee face is not detected in the image

            except:

                self.warning.setVisible(True)
                self.success.setVisible(False)
        # if no employee is selected

        else:
            self.warning.setVisible(True)
        self.filename.setText('')

    # function to browse the file we want

    def browsefiles(self):
        self.success.setVisible(False)
        self.fname = QFileDialog.getOpenFileName(self, 'open file', r'C:', 'JPG Files (*.jpg)')
        self.warning.setVisible(False)
        self.filename.setText(self.fname[0])

    # function to perform cancel operation

    def cancel_add(self):
        if self.filename.text() == '':
            self.frame2.setVisible(False)
        else:
            self.filename.setText('')

# Class for Delete Employee Module


class DeleteEmployee(QDialog):

    # Self function for opening Delete Employee GUI

    def __init__(self):
        super(DeleteEmployee, self).__init__()
        loadUi("GUI/delete_student.ui", self)
        self.back_btn.clicked.connect(HomePage.home_page)
        self.id.setPlaceholderText("18731A0501")
        self.id.setMaxLength(10)
        self.deletemsg.setVisible(False)
        self.deletemsg_2.setVisible(False)
        self.deletebtn.clicked.connect(self.deleteemployee)

    # function to perform delete employee operation

    def deleteemployee(self):
        list1 = TakeAttendence()
        emp_id = self.id.text() + '.jpg'
        # print(list1.myList)
        txt = self.id.text()

        # cheacking for employee is there in the data or not

        if emp_id in list1.myList:
            a_file = open("docs/encoded_data.txt", "r")  # get list of lines.
            lines = a_file.readlines()
            # a_file.
            new_file = open("docs/encoded_data.txt", "w")
            for line in lines:
                if emp_id not in line:  # Delete "line2" from new_file.
                    new_file.write(line)

            path = 'ImagesAttendance/'
            os.remove(path+emp_id)

            d_msg = txt + " has been removed successfully"
            self.deletemsg_2.setVisible(False)
            self.deletemsg.setVisible(True)
            self.deletemsg.setText(d_msg)
            self.id.setText("")

        # if user is not there in the data

        else:
            self.deletemsg_2.setVisible(True)
            d_msg = txt + " was not found"
            self.deletemsg_2.setText(d_msg)

# Class for Create Account Module


class CreateAcc(QDialog):

    # Self function for opening Create Account GUI

    def __init__(self):
        super(CreateAcc, self).__init__()
        loadUi("GUI/createacc.ui", self)
        self.samepass.setVisible(False)
        self.samepass_2.setVisible(False)
        self.signupbutton.clicked.connect(self.createaccfunction)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirmpass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.cancel.clicked.connect(self.loginpage)

    # function to go for login page

    def loginpage(self):
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    # function to perform create accounty operation

    def createaccfunction(self):
        userid = self.userid.text()
        # if no userid is given

        if userid == '':
            createacc = CreateAcc()
            widget.addWidget(createacc)
            widget.setCurrentIndex(widget.currentIndex() + 1)
        # if user id is given

        else:
            # cheacking for both passwoed and confirm password are same or not

            if self.password.text() == self.confirmpass.text():
                password = self.password.text()
                userids = []
                with open('docs/mykey.key', 'rb') as m:
                    key = m.read()
                fer = Fernet(key)
                with open('docs/users.txt', 'rb') as en:
                    original = en.read()
                originals = fer.decrypt(original)
                with open('docs/users.txt', 'wb') as en:
                    en.write(originals)
                with open("docs/users.txt", 'r') as file:
                    while True:
                        line = file.readline()
                        if not line:
                            break
                        userids.append(line.split()[0])
                if userid not in userids:
                    with open('docs/users.txt', 'a') as f:
                        f.write(userid + " " + password + "\n")
                    with open('docs/users.txt', 'rb') as en:
                        original = en.read()
                    encrypted = fer.encrypt(original)
                    with open('docs/users.txt', 'wb') as en:
                        en.write(encrypted)
                    # if user is added successfully

                    login = Login()
                    widget.addWidget(login)
                    widget.setCurrentIndex(widget.currentIndex() + 1)
                else:
                    self.samepass_2.setVisible(True)
                    self.userid.setText("")
                    self.password.setText("")
                    self.confirmpass.setText("")

            # if password and confirm password are not matched

            else:
                self.samepass.setVisible(True)
                self.userid.setText("")
                self.password.setText("")
                self.confirmpass.setText("")


app = QApplication(sys.argv)
mainwindow = Login()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(900)
widget.setFixedHeight(600)
widget.show()
app.exec_()
