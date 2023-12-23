# Описание оптимизации config.pbtxt для тритонсервера.

## Системные характеристики

|Type|Value|
|-|-|
OS | macOS Ventura. Version 13.3
CPU | Apple M2 Pro
vCPU | 4
RAM | 8

## Описание стуктуры model\_repository

```
├── mnist_cnn - модель для классификации цифр в onnx формате
│   ├── 1 - первая версия
│   │   └── model.onnx - модель в onnx формате
│   └── config.pbtxt - конфиг  модели
```

## Описание решаемой задачи

Данная модель решает задачу классификации чисел из датасета MNIST. Таким образом по входному однокональному изображению 28x28 выдается единственное число, которое представлено на поданном изображении.

(Далее сделаем предположение, что max\_batch\_size в среднем не должен превышать 8. Например, такое значение можно получить из-за ограничений налогаемых заказчиком. Например, на каждом отдельном блоке/предприятии у него 8 регистраторов с которых поступают изображения, а каждый блоке/предприятие отправляет запросы к нашему сервису, например. И количество такиъ блоков/предприятий  планируется не больше 5)

## Experiments

Первый запуск был произведен следующей коммандой:

```sh
perf\_analyzer -m mnist\_cnn -u localhost:8500 - -concurrency-range 1:5 --shape input:16,1,28,28
```


## Метрики
После чего были получены следующие характеристики по latency и Throughput. Данные характеристики представлены в зависимости от размера отправляемых батчей (1, 8, 16):

При этом характеристики config.pbtxt были следующими:

|||
|-|-|
 max\_batch\_size | 8
 max_queue_delay_microseconds | -
 preferred_batch_size | -
 kind | KIND_CPU


 . |  batch=16 | batch=8 |  batch=1
 -|-|-|-|
 C1. Latency99 | 45839 | 47798 | 47172
 C2. Latency99 | 48195 | 48062 | 49387
 C3.  Latency99 | 49647 | 47519 | 50780
 C4. Latency99 | 49549 | 50507 | 51829
 C5. Latency99 | 49436 | 51464 | 52079
 C1. Troughput | 822.623 | 731.956 | 796.877
 C2. Troughput | 857.024 | 879.731 | 913.652
 C3. Troughput | 995.929 | 1059.67 | 1008.6
 C4. Troughput | 1276.72 | 1247.7 | 1116.75
 C5. Troughput | 1448.8 | 1518.22 | 1261.95

Сравнение при batch=8:

```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 731.956 infer/sec, latency 1361 usec
Concurrency: 2, throughput: 879.731 infer/sec, latency 2272 usec
Concurrency: 3, throughput: 1059.67 infer/sec, latency 2830 usec
Concurrency: 4, throughput: 1247.7 infer/sec, latency 3204 usec
Concurrency: 5, throughput: 1518.22 infer/sec, latency 3292 usec
```

Параметры и метрики после оптимизации:

|||
|-|-|
 max\_batch\_size | 8
 max_queue_delay_microseconds | 25
 preferred_batch_size | [4, 8]
 kind | KIND_CPU

 . |  batch=16 | batch=8 |  batch=1
 -|-|-|-|
 C1. Latency99 | 47635 | 46416 |
 C2. Latency99 | 42455 | 44233 |
 C3.  Latency99 | 48479 | 44499 |
 C4. Latency99 | 48480 | 47080 |
 C5. Latency99 | 51075 | 46626 |
 C1. Troughput | 727.931 | 777.198 |
 C2. Troughput | 987.823 | 1033.5 |
 C3. Troughput | 1207.12 | 1334.66 |
 C4. Troughput | 1494.45 | 1457.67 |
 C5. Troughput | 1624.15 | 1718.25 |

Сравнение при batch=8
```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 777.198 infer/sec, latency 1286 usec
Concurrency: 2, throughput: 1033.5 infer/sec, latency 1934 usec
Concurrency: 3, throughput: 1334.66 infer/sec, latency 2247 usec
Concurrency: 4, throughput: 1457.67 infer/sec, latency 2743 usec
Concurrency: 5, throughput: 1718.25 infer/sec, latency 2910 usec
```


## Объяснение мотивации выбора

Стоит понимать, что эксперименты с сервером, использующим onnx модель, проводились на mac, что не давало возможности проводить оптимизацию с использованием gpu, поэтому данный параметр не изменялся.

Во-первых были проведены тесты и изменением ```max_queue_delay_microseconds```.
При увеличении этого параметра до 500 среднии метрики ухудшились, а также значительно ухудшились метрики по Latency p99 при условии batch\_size=8. Было сделано предположение, что лучше уменьшит дилей. Такие же результаты были, когда ```max_queue_delay_microseconds``` устанавливался равным 100. Исходя из этого итог был таков, что увеличение ```max_queue_delay_microseconds``` приводит к ухудшением метрик. При значении ```max_queue_delay_microseconds``` = 25 значения оказались наиболее хорошими из исследуемых. При этом, если полность выключать max_queue_delay_microseconds, то метрики также ухудшаются.

Также были проведены эксперименты с изменением количество элементов подаваемых в батч из тех, что скопились в очереди. Получилось, что если использовать preferred_batch_size [4, 8] или [6, 8], то значения заметно улучшаются при условии, что если мы берем [4, 8], то результаты чуть-чуть лучше. Остальные подходы, когда бралось [2, 8] или [2, 4, 8] показали, что улучшений в метриках никаких нет
