from kafka import KafkaConsumer
def consumer_init(topic_name):
    consumer = KafkaConsumer(topic_name)
    return consumer

