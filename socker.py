import socket
import binascii

# Define o host e a porta em que o servidor irá escutar
HOST = '192.168.0.100'  # Endereço IP local
PORT = 5653        # Porta de escuta

# Cria um socket TCP/IP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Associa o socket com o endereço e a porta
server_socket.bind((HOST, PORT))

# Coloca o socket em modo de escuta
server_socket.listen(10)

print('Aguardando conexões...')

# Aceita uma conexão quando ela chegar
connection, client_address = server_socket.accept()

try:
    print('Conexão recebida de', client_address)
    def convert_to_utf8(byte_str):
    # Decodifica a string de bytes usando 'unicode_escape' e ignora os erros
        utf8_str = byte_str.decode('unicode_escape', errors='ignore')
        return utf8_str

    # Loop para receber dados da conexão
    while True:
        connection, client_address = server_socket.accept()
        print('Conexão recebida de', client_address)
        data = connection.recv(1024)  # Recebe os dados da conexão
        data.replace('b', "").replace("'", "")
        print('[DATA]', data)
        print('[str]', binascii.hexlify(data))

        # if data:
        #     print('Dados recebidos:', utf8_str, data)  # Decodifica os dados recebidos
        
except:
    print("Erro")

finally:
    # Encerra a conexão
    connection.close()
