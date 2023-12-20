if __name__ == "__main__":
    import cv2
    import os

    pasta_origem = 'cards'

    pasta_destino = 'half_cards'

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    for arquivo in os.listdir(pasta_origem):
        if arquivo.endswith(('.png', '.jpg', '.jpeg')):  # Verificar se é uma imagem
            caminho_imagem = os.path.join(pasta_origem, arquivo)
            imagem = cv2.imread(caminho_imagem)
            altura, largura = imagem.shape[:2]
            imagem_metade = imagem[0:int(altura / 2), 0:largura]

            cv2.imwrite(os.path.join(pasta_destino, arquivo), imagem_metade)

    print("Processamento concluído.")
