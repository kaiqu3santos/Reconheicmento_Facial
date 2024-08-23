import os
import face_recognition
import pickle

def train_model():
    user_images_dir = 'user_images'
    known_encodings = []
    known_ids = []

    for user_id in os.listdir(user_images_dir):
        user_folder = os.path.join(user_images_dir, user_id)
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_ids.append(user_id)
                #print(encodings[0])

    # Salve as codificações e os IDs em um arquivo
    with open(r'.\modelos\trained_model.pkl', 'wb') as model_file:
        pickle.dump({'encodings': known_encodings, 'ids': known_ids}, model_file)
    
    print("Modelo treinado e salvo como 'trained_model.pkl'.")

if __name__ == "__main__":
    train_model()
