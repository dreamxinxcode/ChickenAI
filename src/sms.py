from twilio.rest import Client
from twilio.base.exceptions import TwilioException
from decouple import config


class SMS:
    def __init__(self) -> None:
        self.account_sid = config('TWILIO_SID')
        self.auth_token = config('TWILIO_AUTH_TOKEN')
        self.twilio_number = config('TWILIO_NUMBER')
        self.recipients = config('RECIPIENTS', default='').split(',')
        self.client = Client(self.account_sid, self.auth_token)

    def send_sms(self, recipient: str, msg: str) -> None:
        try:
            print(f'Sending SMS to {recipient} - "{message}"')
            message = self.client.messages.create(
                body=message,
                from_=self.twilio_number,
                to=recipient
            )
            print(message.sid)
        except TwilioException as e:
            print('Error sending SMS:', str(e))