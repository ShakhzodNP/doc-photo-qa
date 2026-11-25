from enum import StrEnum


class Reason(StrEnum):
    NO_DOCUMENT = "NO_DOCUMENT"
    SCREENSHOT = "SCREENSHOT"
    BLANK_PAGE = "BLANK_PAGE"
    DOCUMENT_CUT = "DOCUMENT_CUT"
    TOO_SMALL = "TOO_SMALL"
    TOO_TILTED = "TOO_TILTED"
    BLURRY = "BLURRY"
    TOO_DARK = "TOO_DARK"
    OVEREXPOSED = "OVEREXPOSED"
    GLARE = "GLARE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    RESOLUTION_TOO_HIGH = "RESOLUTION_TOO_HIGH"
    INVALID_IMAGE = "INVALID_IMAGE"

    def __repr__(self) -> str:
        return self.value


REASON_MESSAGES = {
    Reason.NO_DOCUMENT: "На фото не найден документ. Сфотографируйте документ целиком.",
    Reason.SCREENSHOT: "Похоже на снимок экрана. Сфотографируйте сам документ камерой телефона.",
    Reason.BLANK_PAGE: "На листе не видно текста. Сфотографируйте сторону документа с содержимым.",
    Reason.DOCUMENT_CUT: "Документ виден не целиком. Отодвиньте телефон, чтобы в кадр попали все четыре угла.",
    Reason.TOO_SMALL: "Документ слишком мелкий в кадре. Поднесите телефон ближе.",
    Reason.TOO_TILTED: "Документ снят под слишком острым углом. Держите телефон параллельно документу.",
    Reason.BLURRY: "Фото размыто. Держите телефон неподвижно и дождитесь фокусировки.",
    Reason.TOO_DARK: "Фото слишком тёмное. Сфотографируйте при лучшем освещении.",
    Reason.OVEREXPOSED: "Фото пересвечено. Уберите яркий свет или выключите вспышку.",
    Reason.GLARE: "На документе блик. Измените угол съёмки или уберите источник света.",
    Reason.FILE_TOO_LARGE: "Файл слишком большой. Загрузите фото до 10 МБ.",
    Reason.RESOLUTION_TOO_HIGH: "Разрешение фото слишком большое. Уменьшите его или сделайте снимок обычной камерой телефона.",
    Reason.INVALID_IMAGE: "Не удалось открыть файл. Загрузите фото в формате JPG или PNG.",
}
