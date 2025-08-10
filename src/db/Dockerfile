FROM alpine
RUN apk add --no-cache sqlite

WORKDIR /app

COPY init.sql ./init.sql
COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x ./entrypoint.sh

CMD ["sh", "entrypoint.sh"]