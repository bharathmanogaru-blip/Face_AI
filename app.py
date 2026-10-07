import os
import json
import uuid
import time
import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory
)

from insightface.app import FaceAnalysis


app = Flask(__name__)


REGISTERED_FOLDER = "registered_members"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
DATA_FILE = "members.json"

MATCH_THRESHOLD = 0.35


os.makedirs(
    REGISTERED_FOLDER,
    exist_ok=True
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


if not os.path.exists(DATA_FILE):

    with open(
        DATA_FILE,
        "w"
    ) as file:

        json.dump(
            [],
            file,
            indent=4
        )


face_app = FaceAnalysis(
    name="buffalo_l"
)

face_app.prepare(
    ctx_id=0,
    det_size=(1280, 1280)
)


def load_members():

    try:

        with open(
            DATA_FILE,
            "r"
        ) as file:

            return json.load(file)

    except:

        return []


def save_members(members):

    with open(
        DATA_FILE,
        "w"
    ) as file:

        json.dump(
            members,
            file,
            indent=4
        )


def create_embedding(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return None

    faces = face_app.get(
        image
    )

    if len(faces) == 0:
        return None

    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        return None

    embedding = (
        embedding / norm
    )

    return embedding


def get_known_faces():

    members = load_members()

    known_faces = []

    for member in members:

        image_path = os.path.join(
            REGISTERED_FOLDER,
            member["filename"]
        )

        if not os.path.exists(
            image_path
        ):
            continue

        embedding = create_embedding(
            image_path
        )

        if embedding is None:
            continue

        known_faces.append(
            {
                "id": member["id"],
                "name": member["name"],
                "filename": member["filename"],
                "embedding": embedding
            }
        )

    return known_faces


def get_member_list():

    members = load_members()

    member_list = []

    for member in members:

        member_list.append(
            {
                "id": member["id"],
                "name": member["name"],
                "photo": url_for(
                    "member_file",
                    filename=member["filename"]
                )
            }
        )

    return member_list


@app.route("/")
def home():

    members = get_member_list()

    return render_template(
        "index.html",
        members=members,
        results=None,
        result_image=None
    )


@app.route(
    "/register-member",
    methods=["POST"]
)
def register_member():

    name = request.form.get(
        "member_name",
        ""
    ).strip()

    photo = request.files.get(
        "member_photo"
    )

    if not name:
        return redirect(
            url_for("home")
        )

    if not photo:
        return redirect(
            url_for("home")
        )

    if photo.filename == "":
        return redirect(
            url_for("home")
        )

    members = load_members()

    for member in members:

        if (
            member["name"].lower()
            == name.lower()
        ):

            return redirect(
                url_for("home")
            )

    extension = os.path.splitext(
        photo.filename
    )[1].lower()

    allowed_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]

    if extension not in allowed_extensions:

        return redirect(
            url_for("home")
        )

    member_id = uuid.uuid4().hex

    filename = (
        member_id +
        extension
    )

    image_path = os.path.join(
        REGISTERED_FOLDER,
        filename
    )

    photo.save(
        image_path
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        os.remove(
            image_path
        )

        return redirect(
            url_for("home")
        )

    faces = face_app.get(
        image
    )

    if len(faces) == 0:

        os.remove(
            image_path
        )

        return redirect(
            url_for("home")
        )

    members.append(
        {
            "id": member_id,
            "name": name,
            "filename": filename
        }
    )

    save_members(
        members
    )

    return redirect(
        url_for("home")
    )


@app.route(
    "/edit-member/<member_id>",
    methods=["POST"]
)
def edit_member(member_id):

    members = load_members()

    selected_member = None

    for member in members:

        if member["id"] == member_id:

            selected_member = member

            break

    if selected_member is None:

        return redirect(
            url_for("home")
        )

    new_name = request.form.get(
        "member_name",
        ""
    ).strip()

    if new_name:

        selected_member[
            "name"
        ] = new_name

    photo = request.files.get(
        "member_photo"
    )

    if photo and photo.filename != "":

        extension = os.path.splitext(
            photo.filename
        )[1].lower()

        allowed_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ]

        if extension in allowed_extensions:

            temp_filename = (
                "temp_" +
                member_id +
                extension
            )

            temp_path = os.path.join(
                REGISTERED_FOLDER,
                temp_filename
            )

            photo.save(
                temp_path
            )

            image = cv2.imread(
                temp_path
            )

            valid_photo = False

            if image is not None:

                faces = face_app.get(
                    image
                )

                if len(faces) > 0:

                    valid_photo = True

            if valid_photo:

                old_path = os.path.join(
                    REGISTERED_FOLDER,
                    selected_member[
                        "filename"
                    ]
                )

                if os.path.exists(
                    old_path
                ):

                    os.remove(
                        old_path
                    )

                new_filename = (
                    member_id +
                    extension
                )

                new_path = os.path.join(
                    REGISTERED_FOLDER,
                    new_filename
                )

                os.rename(
                    temp_path,
                    new_path
                )

                selected_member[
                    "filename"
                ] = new_filename

            else:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )

    save_members(
        members
    )

    return redirect(
        url_for("home")
    )


@app.route(
    "/delete-member/<member_id>",
    methods=["POST"]
)
def delete_member(member_id):

    members = load_members()

    remaining_members = []

    for member in members:

        if member["id"] == member_id:

            image_path = os.path.join(
                REGISTERED_FOLDER,
                member["filename"]
            )

            if os.path.exists(
                image_path
            ):

                os.remove(
                    image_path
                )

        else:

            remaining_members.append(
                member
            )

    save_members(
        remaining_members
    )

    return redirect(
        url_for("home")
    )


@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    file = request.files.get(
        "group_image"
    )

    if not file:

        return redirect(
            url_for("home")
        )

    if file.filename == "":

        return redirect(
            url_for("home")
        )

    input_path = os.path.join(
        UPLOAD_FOLDER,
        "group.jpg"
    )

    result_path = os.path.join(
        RESULT_FOLDER,
        "result.jpg"
    )

    file.save(
        input_path
    )

    group = cv2.imread(
        input_path
    )

    if group is None:

        return redirect(
            url_for("home")
        )

    for filename in os.listdir(
        RESULT_FOLDER
    ):

        if filename.startswith(
            "detected_"
        ):

            old_file = os.path.join(
                RESULT_FOLDER,
                filename
            )

            try:

                os.remove(
                    old_file
                )

            except:

                pass

    faces = face_app.get(
        group
    )

    known_faces = get_known_faces()

    results = []

    for face_number, face in enumerate(
        faces,
        start=1
    ):

        face_start_time = time.time()

        embedding = face.embedding

        norm = np.linalg.norm(
            embedding
        )

        if norm != 0:

            embedding = (
                embedding / norm
            )

        best_name = "Unknown"
        best_score = 0.0
        best_filename = ""

        for known in known_faces:

            score = float(
                np.dot(
                    embedding,
                    known["embedding"]
                )
            )

            if score > best_score:

                best_score = score

                best_name = known[
                    "name"
                ]

                best_filename = known[
                    "filename"
                ]

        similarity_percent = (
            best_score * 100
        )

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        height, width = group.shape[:2]

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            width - 1,
            x2
        )

        y2 = min(
            height - 1,
            y2
        )

        detected_filename = (
            f"detected_{face_number}.jpg"
        )

        detected_path = os.path.join(
            RESULT_FOLDER,
            detected_filename
        )

        if x2 > x1 and y2 > y1:

            face_crop = group[
                y1:y2,
                x1:x2
            ]

            cv2.imwrite(
                detected_path,
                face_crop
            )

        if best_score >= MATCH_THRESHOLD:

            status = "Matched"

            box_color = (
                0,
                255,
                0
            )

        else:

            best_name = "Unknown"

            best_filename = ""

            status = "Unknown"

            box_color = (
                0,
                0,
                255
            )

        cv2.rectangle(
            group,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        detection_time = round(
            time.time()
            - face_start_time,
            3
        )

        member_photo = ""

        if best_filename:

            member_photo = url_for(
                "member_file",
                filename=best_filename
            )

        results.append(
            {
                "name": best_name,
                "similarity": similarity_percent,
                "status": status,
                "member_photo": member_photo,
                "detected_face": detected_filename,
                "detection_time": detection_time
            }
        )

    cv2.imwrite(
        result_path,
        group
    )

    members = get_member_list()

    return render_template(
        "index.html",
        members=members,
        results=results,
        result_image="result.jpg"
    )


@app.route(
    "/registered-members/<filename>"
)
def member_file(filename):

    return send_from_directory(
        REGISTERED_FOLDER,
        filename
    )


@app.route(
    "/results/<filename>"
)
def result_file(filename):

    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )