# **Как пользоваться** (инструкция для macOS)

# **Предварительные действия**

## 1. **Установите ollama**:
### https://ollama.com/download
## 2. **Запулльте модель**:
```
ollama pull "llama3.2:3b"
```
## 3. **Проверьте, установлены ли необходимые пекэджи**:
### **Список либ**:
#### flask, ollama, langchain
## 4. **Запустите flash_server.py**

# **Интерфейс** (с помощью cURL):

## **Инициализация пользователя**:
```shell
curl -X POST http://127.0.0.1:8000/LLM/initialize -H 'Content-Type: application/json' -d '{"user_id":"<user_id>"}'
```
## **Вопрос ЛЛМке**:
```shell
curl -X POST http://127.0.0.1:8000/LLM/chat -H 'Content-Type: application/json' -d '{"user_id":"<user_id>","message":"<message>"}'
```
## **Вопрос ЛЛМке вместе с файлом** (сейчас доступ к файлу только через путь в директории)
```shell
curl -X POST http://127.0.0.1:8000/LLM/chat -H 'Content-Type: application/json' -d '{"user_id":"<user_id>","message":"<message>","file_path":"<file_path>"}'
```
## **Очистить контекст**:
```shell
curl -X POST http://127.0.0.1:8000/LLM/clear -H 'Content-Type: application/json' -d '{"user_id":"<user_id>"}'```
