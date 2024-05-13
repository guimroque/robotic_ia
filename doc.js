/*

    [INTRO]

        Os próximos subtópicos são destinados a explicar as partes componentes do sistema de controle utilizando visão computacional proposto.
        Este trabalho busca propor uma maneira de controlar um sistema de forma autônoma, utilizando visão computacional para a identificação 
        de objetos e a tomada de decisões.
        Foram utilizados itens do laboratório de automação do Instituto Nacional de Telecomunicações (Inatel) para a construção do sistema.


    [COMPONENTES]
        [BLOCOS]
            Blocos de plástico impressos em 3D, nas cores branca, preta e cinza, com dimensões de [NN]cm x [NN]cm x [NN]cm,
            foram empregados como modelos físicos dos objetos manipulados durante as atividades de teste e desenvolvimento do sistema robótico.
            [REFERENCIA IMAGEM]

        [MESA DE TRABALHO]
            A mesa de trabalho, feita de madeira e com dimensões de [NN]cm x [NN]cm x [NN]cm, proporciona uma superfície plana 
            para a movimentação tanto dos blocos quanto do robô durante as atividades de manipulação.
            [REFERENCIA IMAGEM]

        [CÂMERA]
            A captura de imagens é realizada pela câmera [ESPECIFICACAO_CAMERA], com resolução de [NN]x[NN] pixels e taxa de quadros de [NN] fps, 
            está fixada por uma haste acima do robô, a 1.04[m] acima da mesa de trabalho, e é responsável por capturar as imagens e transmiti-las
            ao PC para processamento.
            [REFERENCIA IMAGEM]

        [ROBÔ]
            O robô utilizado, o [ESPECIFICACAO_ROBO], posicionado sobre a mesa de trabalho, é equipado com uma garra pneumática 
            de movimento de pinça [ESPEFICACAO_MODELO], controlada pelo [ESPECIFICACAO_CONTROLADOR], que recebe comandos de movimento e 
            controle da garra. Este controlador utiliza a linguagem VAL3 para a elaboração de programas de movimento e controle do robô, 
            estando conectado a um PC [ESPECIFICACOES] para execução do algoritmo de controle.
            [REFERENCIA IMAGEM]

        [MODELO DE DETECÇÃO DE OBJETOS]
            Para a detecção de objetos, foi utilizado o avançado algoritmo YOLO (You Only Look Once) [DETALHES_VERSAO] [REFERENCIA], 
            que, ao processar a imagem em uma única passagem, prevê diretamente as caixas delimitadoras e classes dos objetos, 
            destacando-se por sua eficiência e precisão. O modelo foi treinado com um dataset próprio, construído a partir de [NN] 
            imagens capturadas pela câmera [REFERENCIA AO ITEM ROBO], posicionada por uma haste fixa acima do robô e a 1.04[m] acima da mesa de trabalho, 
            e rotuladas manualmente utilizando o Roboflow [REFERENCIA]. 
            O treinamento do modelo foi realizado em um PC [ESPECIFICACOES] com os seguintes parâmetros: [NN] épocas, 
            [NN] imagens e 4 classes (Bloco branco, Bloco preto, Bloco cinza e Mesa de trabalho).

        [MESA DE TRABALHO VIRTUAL]
            Foi desenvolvida uma réplica virtual da mesa de trabalho, utilizando o software de simulação RoboDK [REFERENCIA],
            com o objetivo de fazer deste um gêmeo digital do ambiente real.
            A mesa virtual possui as mesmas dimensões da mesa real e é composta por um plano de trabalho, blocos virtuais,
            câmera virtual e robô virtual, que são modelados de acordo com as especificações dos componentes reais.
            [REFERENCIA_IMAGEM]

        [ALGORITMO DE CONTROLE]
            O algoritmo de controle, desenvolvido em Python, desempenha o papel fundamental de processar as imagens capturadas pela câmera, 
            realizar a identificação dos objetos presentes na cena e tomar decisões de movimento para o robô. Para facilitar a execução do algoritmo, 
            foram criadas bibliotecas específicas, como Cam, Frame, Object e Network, que são responsáveis por tarefas como captura de imagem, 
            identificação de objetos e suas posições, posicionamento dos blocos de forma simétrica em relação aos blocos físicos no ambiente simulado e 
            estabelecimento de comunicação via socket com o controlador do robô. Essas bibliotecas foram desenvolvidas com o intuito de 
            otimizar o processo de controle do robô durante as operações de manipulação de objetos.
        
        [IMAGEM]
            A figura abaixo ilustra o sistema de controle proposto, com os componentes descritos anteriormente.


        


        







*/