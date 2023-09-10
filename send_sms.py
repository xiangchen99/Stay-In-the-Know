# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ['TWILIO_ACCOUNT_SID'] = 'AC10b029395650ed8afd05108e47968825'
auth_token = os.environ['TWILIO_AUTH_TOKEN'] = '6554affbc745cb90bd319d60cd450bd1'
client = Client(account_sid, auth_token)

message = client.messages.create(
                              from_='+18883015401',
                              body='Hi there',
                              to='+16463227786'
                          )

print(message.sid)
