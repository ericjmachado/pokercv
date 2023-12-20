import cv2

from scripts import get_qtd_usuarios


def get_minha_mao(imagem):

    template = cv2.imread('./images/pattern_user_2.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    altura_t, largura_t = template.shape[:2]

    resultado = cv2.matchTemplate(imagem, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)

    top_left = max_loc

    bottom_right = (top_left[0] + largura_t, top_left[1] + altura_t)

    x1 = max(top_left[0] - 300, 0)
    y1 = max(top_left[1] - 300, 0)
    x2 = min(bottom_right[0] + 300, imagem.shape[1])
    y2 = min(bottom_right[1] + 300, imagem.shape[0])


    imagem_recortada = imagem[y1:y2, x1:x2]

    cv2.imshow('Imagem Original', imagem)
    cv2.imshow('Imagem Recortada', imagem_recortada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def metodo():
    image = cv2.imread('images/image1.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    usuarios, image = get_qtd_usuarios(image)
    cv2.imshow(f'Usuarios {usuarios}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    metodo()
