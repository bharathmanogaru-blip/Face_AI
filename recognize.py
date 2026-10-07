import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

KNOWN_FOLDER = "known_faces"
GROUP_IMAGE = "group.jpg"
RESULT_IMAGE = "result.jpg"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(1280, 1280))

known_faces = []

for person_name in os.listdir(KNOWN_FOLDER):

    person_folder = os.path.join(KNOWN_FOLDER, person_name)

    if not os.path.isdir(person_folder):
        continue

    for file_name in os.listdir(person_folder):

        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(person_folder, file_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"No face found: {person_name}/{file_name}")
            continue

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) *
                          (f.bbox[3] - f.bbox[1])
        )

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        known_faces.append({
            "name": person_name,
            "embedding": embedding
        })

        print(f"Registered: {person_name} - {file_name}")

if len(known_faces) == 0:
    print("No registered faces found.")
    exit()

group = cv2.imread(GROUP_IMAGE)

if group is None:
    print(f"Cannot open {GROUP_IMAGE}")
    exit()

faces = app.get(group)

print(f"\nFaces detected: {len(faces)}\n")

for face_number, face in enumerate(faces, start=1):
    embedding = face.embedding
    embedding = embedding / np.linalg.norm(embedding)

    best_name = "Unknown"
    best_score = -1

    for known in known_faces:
        score = np.dot(embedding, known["embedding"])

        if score > best_score:
            best_score = score
            best_name = known["name"]

    x1, y1, x2, y2 = map(int, face.bbox)

    if best_score >= 0.35:
        box_color = (0, 255, 0)

        cv2.rectangle(group, (x1, y1), (x2, y2), box_color, 3)

        cv2.putText(
            group,
            best_name,
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            box_color,
            3
        )

        print(f"Face {face_number}: {best_name} | Similarity: {best_score:.4f}")

    else:
        box_color = (0, 0, 255)

        cv2.rectangle(group, (x1, y1), (x2, y2), box_color, 3)

        print(f"Face {face_number}: Unknown | Similarity: {best_score:.4f}")

cv2.imwrite(RESULT_IMAGE, group)

cv2.namedWindow(
    "Face Recognition Result",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Face Recognition Result",
    1000,
    700
)

cv2.imshow(
    "Face Recognition Result",
    group
)

print("\nResult saved as result.jpg")

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()