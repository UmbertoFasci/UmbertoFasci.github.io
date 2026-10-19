-- {{< streamlit https://your-app.streamlit.app height="640" title="Forecast explorer" caption="Drag the slider to change the horizon." >}}
-- Embeds a Streamlit app in a branded frame. Adds Streamlit's embed flags so the
-- app drops its toolbar and stays in the light theme.

local function str(v, default)
  if v == nil then return default end
  local s = pandoc.utils.stringify(v)
  if s == "" then return default end
  return s
end

local function esc(s)
  return (s:gsub("&", "&amp;"):gsub('"', "&quot;"):gsub("<", "&lt;"):gsub(">", "&gt;"))
end

return {
  ["streamlit"] = function(args, kwargs)
    local url = str(args[1], nil)
    if url == nil then
      return pandoc.Strong(pandoc.Str("[streamlit shortcode: missing app URL]"))
    end
    local height = str(kwargs["height"], "640")
    local title = str(kwargs["title"], "Interactive Streamlit app")
    local caption = str(kwargs["caption"], "")
    local sep = url:find("?", 1, true) and "&" or "?"
    local src = url .. sep .. "embed=true&embed_options=light_theme"

    local html = string.format(
      '<figure class="app-embed">' ..
      '<iframe src="%s" title="%s" height="%s" loading="lazy" allow="clipboard-write"></iframe>' ..
      '%s' ..
      '<a class="app-embed-open" href="%s" target="_blank" rel="noopener">Open the app in a new tab</a>' ..
      '</figure>',
      esc(src), esc(title), esc(height),
      caption ~= "" and ('<figcaption>' .. esc(caption) .. '</figcaption>') or "",
      esc(url))
    return pandoc.RawBlock("html", html)
  end
}
