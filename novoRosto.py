import cv2
import os

def capture_photos(user_id, num_photos=50):
    # Crie uma pasta para armazenar as imagens do usuário
    user_folder = f"user_images/{user_id}"
    os.makedirs(user_folder, exist_ok=True)

    # Iniciar a captura de vídeo
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao acessar a câmera.")
            break

        cv2.imshow('Capturing Photos', frame)
        cv2.imwrite(os.path.join(user_folder, f"{user_id}_{count}.jpg"), frame)
        count += 1
        print(count)

        # Aguarde 100 ms entre as capturas
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captura de {num_photos} fotos concluída e salva na pasta {user_folder}.")

if __name__ == "__main__":
    user_id = input("Digite o ID do usuário: ")
    capture_photos(user_id)
