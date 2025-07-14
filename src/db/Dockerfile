FROM alpine
RUN apk add --no-cache sqlite
COPY init.sql /init.sql
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
