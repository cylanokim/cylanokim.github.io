FROM docker.elastic.co/elasticsearch/elasticsearch:8.17.4

# nori analyzer 플러그인 설치 
RUN bin/elasticsearch-plugin install analysis-nori
