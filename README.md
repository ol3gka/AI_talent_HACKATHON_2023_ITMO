# AI_talent_HACKATHON_2023_ITMO
В этом репозитории представлены материалы по AI_talent_HACKATHON_2023, проходящему в ИТМО в сентябре 2023

Задача хакатона состояла в создании Physics informed модели машинного обучения, которая на основе косвенных данных с датчиков будет предсказывать выбросы CO. Для демонстрации работы модели создан пользовательский интерфейс.

Решение представляет из себя две части

```1) Построение ML пайплайна ``` 
<!-- #region -->
<p align="center">
<img  src="pictures/1.png" width="200">
  <img  src="pictures/2.png">
</p>

Подход на основе lightgbm и Optuna, обученная модель в формате pickle в дальнейшем передается в графический итерфейс для проведения расчета

```2) Построение API, графического интерфейса для экологов для контроля выбросов в прямом времени```

Подсчитывает метрики и выводит результаты

<!-- #region -->
<p align="center">
<img  src="pictures/Скриншот 09-09-2023 00.00.43.png">
</p>

Состав команды:

Николаев Олег, ИТМО

Степочкина Анна, ИТМО

Какуркина Дарья, АГНИ

Люосева Елизавета, ИТМО
