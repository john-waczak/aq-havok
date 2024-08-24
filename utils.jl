function parse_datetime(dt, tz)
    dt_out = String(dt)
    dt_out = ZonedDateTime(DateTime(split(dt, "Z")[1]), tz"UTC")
    return  astimezone(dt_out, tz)
end



