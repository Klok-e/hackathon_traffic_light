# hackathon_traffic_light

## Тема: обнаружение проезда на красный свет
Детекция происходит в несколько этапов:
1. YOLO для всех объектов на картинке и выделить только машины
2. Найти траекторию движения машин с помощью какой-то математики (https://arxiv.org/pdf/1602.00763.pdf)
3. Проверить пересечения траекторий машин и линий дороги перед светофором

Обнаруженные машины нужно как-то отобразить пользователю (для целей демонстрации).

Ещё было бы неплохо по фоткам нарушивших машин определять номер автомобиля.

## Реализовано
- [x] Обнаружение машин на видео
- [x] Нахождение траектории движения машин (SORT)
- [ ] Нахождение траектории машин (Deep SORT)
- [x] Проверка нарушения по траектории и линиям светофора
- [ ] Автоматическое нахождение линий светофора (но чтобы можно было поправить вручную)
- [x] Пользовательский интерфейс в виде веб страницы (flask)
- - [x] На странице стрим видео с обнаруженными машинами ([вот вдохновение](https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/))
- - [x] Ручная установка линий светофора
- - [x] Одна стоп линия с кнопкой переключения _красный/зелёный_
- - [ ] Список стоп диний светофора с кнопками выбора _красный/зелёный_
- - [ ] Список номеров автомобилей, нарушивших правила
