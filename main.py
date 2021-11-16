import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#cap = cv2.VideoCapture('Videos/3.mp4')
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

nf=40   #number of frames
cf=0    #current frame
Tval=1000
seg1=[]
seg2=[]
left=[12,14,16,26,28]
right=[11,13,15,25,27]
num_mo = 20 #motion limit for 2 mins
mo_count = 0
state = "DETECTING..!!!"
st = "DETECTING..!!!"
pTime = 0
frames=0
while True:
    success, img = cap.read()
    width , height = 1280, 720
    img= cv2.resize(img,(width , height))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    lmList = []
    right_val=0
    left_val=0
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        if len(lmList) != 0:
            for i in left:
                cv2.circle(img, (lmList[i][1], lmList[i][2]), 10, (0, 0, 255), cv2.FILLED)
                left_val += lmList[i][1]

            for i in right:
                cv2.circle(img, (lmList[i][1], lmList[i][2]), 10, (0, 0, 255), cv2.FILLED)
                right_val += lmList[i][1]

            val=right_val-left_val

            if cf < nf/2:
                seg1.append(val)
                cf += 1
            elif cf < nf:
                seg2.append(val)
                cf += 1
            else:
                seg = sum(seg1)-sum(seg2)
                if abs(seg) > Tval:
                    state = "MOTION"
                    mo_count += 1
                else:
                    state = "No MOTION"
                seg1 = seg1[1:]
                seg1.append(seg2[0])
                seg2 = seg2[1:]
                seg2.append(val)
                cf = nf
            frames += 1
            print(frames,mo_count)
            if frames >= 80:
                if mo_count >= num_mo:
                    st = "PANIC"
                else:
                    st = "CALM"
                frames=0
                mo_count=0
            else:
                cv2.putText(img, str(st), (170, 150), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 0), 3)

                #print(seg)
            #print(cf,val,"\nseg1=",seg1,"\nseg2=",seg2)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.putText(img, state, (170, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(1)