import clickhouse_connect
import torch
from scipy.spatial import distance
from transformers import AutoModel, AutoTokenizer, pipeline, Conversation
from typing import List, Dict
from server.config import (
    TABLE_NAME,
    HOST,
    PORT,
    MODEL_EMB_NAME,
    MODEL_CHAT_NAME,
    SYSTEM_PROMPT,
)


def search_results(connection, table_name: str, vector: list[float], limit: int = 5):
    """
    Поиск результатов похожих векторов в базе данных.

    Parameters:
    - connection (Connection): Соединение с базой данных.
    - table_name (str): Название таблицы, содержащей вектора и другие данные.
    - vector (List[float]): Вектор для сравнения.
    - limit (int): Максимальное количество результатов.

    Returns:
    - List[dict]: Список результатов с наименованием, URL, датой, номером, текстом и расстоянием.

    Examples:
    >>> connection = Connection(...)
    >>> vector = [0.1, 0.2, 0.3]
    >>> results = search_results(connection, 'my_table', vector, limit=5)
    """
    # Инициализируем список результатов
    res = []

    # Выполняем запрос к базе данных
    with connection.query(f"SELECT * FROM {table_name}").rows_stream as stream:
        for item in stream:
            name, url, date, num, text, emb = item

            # Вычисляем косинусное расстояние между векторами
            dist = distance.cosine(vector, emb)

            # Добавляем результат в список
            res.append(
                {
                    "name": name,
                    "url": url,
                    "date": date,
                    "num": num,
                    "text": text,
                    "dist": dist,
                }
            )

    # Сортируем результаты по расстоянию
    res.sort(key=lambda x: x["dist"])

    # Возвращаем первые limit результатов
    return res[:limit]


def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Выполняет усреднение токенов входной последовательности на основе attention mask.

    Parameters:
    - model_output (tuple): Выход модели, включающий токенов эмбеддинги и другие данные.
    - attention_mask (torch.Tensor): Маска внимания для указания значимости токенов.

    Returns:
    - torch.Tensor: Усредненный эмбеддинг.

    Examples:
    >>> embeddings = model_output[0]
    >>> mask = torch.tensor([[1, 1, 1, 0, 0]])
    >>> pooled_embedding = mean_pooling((embeddings,), mask)
    """
    # Получаем эмбеддинги токенов из выхода модели
    token_embeddings = model_output[0]

    # Расширяем маску внимания для умножения с эмбеддингами
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    # Умножаем каждый токен на его маску и суммируем
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

    # Суммируем маски токенов и обрезаем значения, чтобы избежать деления на ноль
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Вычисляем усредненный эмбеддинг
    return sum_embeddings / sum_mask


def txt2embeddings(text: str, tokenizer, model, device: str = "cpu") -> torch.Tensor:
    """
    Преобразует текст в его векторное представление с использованием модели transformer.

    Parameters:
    - text (str): Текст для преобразования в векторное представление.
    - tokenizer: Токенизатор для предобработки текста.
    - model: Модель transformer для преобразования токенов в вектора.
    - device (str): Устройство для вычислений (cpu или cuda).

    Returns:
    - torch.Tensor: Векторное представление текста.

    Examples:
    >>> text = "Пример текста"
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    >>> model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    >>> embeddings = txt2embeddings(text, tokenizer, model, device="cuda")
    """
    # Кодируем входной текст с помощью токенизатора
    encoded_input = tokenizer(
        [text],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    # Перемещаем закодированный ввод на указанное устройство
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Получаем выход модели для закодированного ввода
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Преобразуем выход модели в векторное представление текста
    return mean_pooling(model_output, encoded_input["attention_mask"])


def load_chatbot(model: str):
    """
    Загружает чатбота для указанной модели.

    Parameters:
    - model (str): Название модели для загрузки чатбота.

    Returns:
    - Conversation: Объект чатбота, готовый для использования.

    Examples:
    >>> chatbot = load_chatbot("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    """
    # Загружаем чатбот с помощью pipeline из библиотеки transformers
    chatbot = pipeline(
        model=model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cuda",
        task="conversational",
    )
    return chatbot


def generate_answer(chatbot, chat: List[Dict[str, str]], document: str) -> str:
    """
    Генерирует ответ от чатбота на основе предоставленного чата и, возможно, документа.

    Parameters:
    - chatbot (Conversation): Объект чатбота.
    - chat (List[Dict[str, str]]): Список сообщений в чате.
    - document (str): Документ, если он предоставлен.

    Returns:
    - str: Сгенерированный ответ от чатбота.

    Examples:
    >>> chat = [
    >>>     {"role": "user", "content": "Привет, как дела?"},
    >>>     {"role": "system", "content": "Всё отлично, спасибо!"},
    >>> ]
    >>> document = "Это документ для обработки"
    >>> answer = generate_answer(chatbot, chat, document)
    """
    # Создаем объект разговора
    conversation = Conversation()

    # Добавляем системное приветствие
    conversation.add_message({"role": "system", "content": SYSTEM_PROMPT})

    # Добавляем документ, если он предоставлен
    if document:
        document_template = """
        CONTEXT:
        {document}

        Отвечай только на русском языке.
        ВОПРОС:
        """
        conversation.add_message({"role": "user", "content": document_template})

    # Добавляем сообщения из чата
    for message in chat:
        conversation.add_message(
            {"role": message["role"], "content": message["content"]}
        )

    # Генерируем ответ от чатбота
    conversation = chatbot(
        conversation,
        max_new_tokens=512,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,
        do_sample=True,
        num_beams=5,
        early_stopping=True,
    )

    # Возвращаем последнее сообщение чатбота как ответ
    return conversation[-1]


def load_models(model: str) -> tuple:
    """
    Загружает токенизатор и модель для указанной предобученной модели.

    Parameters:
    - model (str): Название предобученной модели, поддерживаемой библиотекой transformers.

    Returns:
    - tuple: Кортеж из токенизатора и модели.

    Examples:
    >>> tokenizer, model = load_models("ai-forever/sbert_large_nlu_ru")
    """
    # Загружаем токенизатор для модели
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Загружаем модель
    model = AutoModel.from_pretrained(model)

    return tokenizer, model


if __name__ == "__main__":
    # Подключаемся к ClickHouse
    client = clickhouse_connect.get_client(host=HOST, port=PORT)
    print("Ping:", client.ping())

    # Загружаем токенизатор и модель для векторизации текста чата
    tokenizer, model = load_models(MODEL_EMB_NAME)

    # Загружаем чатбота
    chatbot = load_chatbot(MODEL_CHAT_NAME)

    # Задаем текст чата
    chat = [
        {
            "role": "user",
            "content": "Федеральная антимонопольная служба и Центральный банк Российской Федерации рекомендуют кредитным организациям прозрачно информировать клиентов о бонусных программах, включая условия начисления кешбэка, чтобы избежать ввода потребителей в заблуждение и нарушения законодательства о конкуренции и рекламе.",
        }
    ]

    # Получаем векторное представление текста чата
    embeddings = txt2embeddings(chat[0]["content"], tokenizer, model)[0]

    # Ищем наиболее подходящие документы в базе данных ClickHouse
    search = search_results(client, TABLE_NAME, embeddings, 10)
    document = search[0]["text"]

    # Генерируем ответ чатбота
    chatbot_answer = generate_answer(chatbot, chat, document)

    print("ML model unloaded")
