function get_past_forecast(
    lat_lon::Tuple{Float64,Float64},
    measure_names::Vector{Symbol},
    start_date::Date,
    end_date::Date,
    past_days::Vector{Int64};
    # n_days_ahead::Int64=15,
    apikey::Union{Nothing, String}=nothing,
    max_retries::Int64=2, 
    retry_delay::Float64=5.0
)
    # Verify arguments
    (all(map(m -> (m ∈ getfield.(ADMISSIBLE_MEASURES, :name)), measure_names))) ||
        throw(ErrorException("Invalid measure"))
    # (n_past_days ∈ 0:92) || throw(ErrorException("Invalid n_past days"))

    # Construct url
    # base url given if we pay or not
    if isnothing(apikey)
        base_url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    elseif isa(apikey, String)
        base_url = "https://customer-previous-runs-api.open-meteo.com/v1/forecast"
    end
    # Prepare measures
    all_measures_with_past_days = vcat(
        map(
            m -> [(string(m) * "_previous_day" * string(pd)) for pd in past_days],
            measure_names
        )...
    )

    # query prepraration
    query = [
            "latitude" => lat_lon[1],
            "longitude" => lat_lon[2],
            "hourly" => join(all_measures_with_past_days, ','),
            "start_date" => Dates.format(start_date, dateformat"yyyy-mm-dd"),
            "end_date" => Dates.format(end_date, dateformat"yyyy-mm-dd"),
            "timezone" => "GMT",
            # "forecast_days" => n_days_ahead
    ]
    if !isnothing(apikey)
        push!(query, "apikey" => apikey)
    end
    url = HTTP.URIs.URI(
        base_url,
        query = query
    )

    # Query
    res = get_with_retry(url, max_retries=max_retries, delay=retry_delay)

    # Parse output and put into table
    records = JSON.parse(String(res.body))
    if haskey(records, "hourly")
        records["hourly"] = extract_hourly(
            records["hourly"],
            true
        )
    end

    # Prepare units
    units = prepare_hourly_units(records["hourly_units"])

    # Instantiate output
    weath_forecast = Forecasts(
        (records["latitude"], records["longitude"]),
        records["elevation"],
        units,
        records["hourly"],
        is_dt_regular(records["hourly"], :dt_target, [:dt_created], Hour(1))
    )

    return weath_forecast
    # return records["hourly"] 
end
