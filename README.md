# ABSTRACT

SUPPORT MAIL BOT (SMB)

A tool that combines:

# STRUCTURE

## DB

Sqlite database for historical data.
## GPT 

A container with a tokenized mistral model

## MAILER 

A simple py mail client to scan new mail

## ORQUESTRATOR

It sends a response to a given email. It is a simple py app that combines vectorized data with sqlite data to create a response using that and GPT. 

# USAGE

First clone the repository

```
git clone https://github.com/JorgeMartinezPizarro/smb

```

Then, setup your configuration in .env file. 

Finally,

```
make up
```

to start the services.