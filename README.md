# Система для видеоанализа очереди
Данный репозиторий содержит прототип решения для анализа очереди.
Зависимости, которые требуется удовлетворить, используя pip:
* ultralytics (для YOLO)
* PyTorch
* shapely
* numpy
* cv2 (opencv)

## Файлы, представленные в репозитории
* solution.py - скрипт, осуществляющий анализ видеопотока.
* examples/ - директория, в которой представлено использование скрипта на предоставленных видеозаписях.
  * cameras/ - директория, содержащая настройки многоугольников отдельно для двух данных ракурсов камер.
  * videos/ - директория, содержащая предоставленные видео. Видео разделены по ракурсам и находятся в двух директориях cam1/ и cam2/.
  * results/ - директория, содержащая результат анализа предоставленных видео. Видео разделены по ракурсам и находятся в двух директориях cam1/ и cam2/.
  * screenshots/ - директория, содержащая снимки, сделанные во время анализа видео.
