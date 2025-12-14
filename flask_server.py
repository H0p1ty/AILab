from flask import Flask, request, jsonify
from wip import Chatbot


app = Flask(__name__)
LLMs = dict()

@app.route('/LLM/initialize', methods=['POST'])
def LLM_initialize():
    data = request.get_json()
    if "user_id" not in data:
        return jsonify(
            answer=f"No \"user_id\" field in json"
        )
    user_id = data["user_id"]
    if user_id in LLMs:
        return jsonify(
            answer=f"User \"{user_id}\" is already initialized"
        )
    LLMs[user_id] = Chatbot()
    return jsonify(
        user_id=user_id,
        message="LLM initialized"
    )

@app.route('/LLM/clear', methods=['POST'])
def LLM_clear():
    data = request.get_json()
    if "user_id" not in data:
        return jsonify(
            answer=f"No \"user_id\" field in json"
        )
    user_id = data["user_id"]
    if user_id not in LLMs:
        return jsonify(
            answer=f"User \"{user_id}\" wasn't initialized"
        )
    LLMs[user_id].clear_context()
    return jsonify(
        user_id=user_id,
        message="LLM cleared"
    )

@app.route('/LLM/chat', methods=['POST'])
def LLM_chat():
    data = request.get_json()
    if "user_id" not in data:
        return jsonify(
            answer=f"No \"user_id\" field in json"
        )
    user_id = data["user_id"]
    if user_id not in LLMs:
        return jsonify(
            answer=f"User \"{user_id}\" wasn't initialized"
        )
    if "message" not in data:
        return jsonify(
            answer=f"No \"message\" field in json"
        )
    file_path = None
    if "file_path" in data:
        file_path = data["file_path"]
        LLMs[user_id].add_file(file_path)
    message = data["message"]
    answer = LLMs[user_id].ask(message)
    if answer is not None:
        return jsonify(
            message=answer
        )
    else:
        return jsonify(
            message="something went wrong..."
        )

if __name__ == '__main__':
    app.run(port=8000)
    # app.run(host="0.0.0.0", port=5000)
