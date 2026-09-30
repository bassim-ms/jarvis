"""
Devices: which devices Jarvis knows, which one is talking, and whether it is trusted.

THE BOUNDARY (read this before touching the request entry points in app/main.py):

  /chat          The existing LEGACY / LOCAL application interface. It is anonymous: it carries no device
                 identity and is NEVER proof of one. A request arriving on /chat from localhost is not "the
                 trusted PC" - localhost is not authorisation - and nothing on /chat (body, message text, headers,
                 conversation) can make it one. JarvisCore is called exactly as it always has been, with no
                 DeviceContext.
  /device/chat   The AUTHENTICATED device interface. Only this route (through the DeviceGateway, gateway.py)
                 establishes a trusted DeviceContext, from a device_id + credential the registry verifies and whose
                 trust state is TRUSTED. /device/check-in works the same way.

The two are deliberately NOT equivalent, and /chat is deliberately not modified in Slice 1 (existing clients -
including the PC voice client - depend on it). A DeviceContext is identity information for Jarvis's context; it
grants no permission. What may be done is still decided by the ToolRegistry, the capability boundary and the
confirmation rules, whichever device (or none) asks.

KNOWN, INTENTIONAL LIMITATION (Slice 1): the current PC voice client does not send the device credential, so it
still uses /chat. /device/chat is implemented and tested but no shipped client uses it yet; moving the voice client
onto it is a future integration slice.
"""
