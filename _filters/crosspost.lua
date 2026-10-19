-- Adds an "Also published on" line to the end of a post when its front matter has
--   crosspost:
--     medium: https://medium.com/...
--     linkedin: https://www.linkedin.com/pulse/...
-- Your site stays the original; Medium/LinkedIn copies point back here.

local labels = { medium = "Medium", linkedin = "LinkedIn" }
local order = { "medium", "linkedin" }

function Pandoc(doc)
  local cp = doc.meta.crosspost
  if cp == nil or not quarto.doc.is_format("html") then return doc end

  local links = {}
  for _, key in ipairs(order) do
    local v = cp[key]
    if v ~= nil then
      local url = pandoc.utils.stringify(v)
      if url ~= "" then
        table.insert(links, pandoc.Link(labels[key], url))
      end
    end
  end
  if #links == 0 then return doc end

  local inlines = { pandoc.Str("Also published on ") }
  for i, l in ipairs(links) do
    if i > 1 then
      table.insert(inlines, pandoc.Str(i == #links and " and " or ", "))
    end
    table.insert(inlines, l)
  end
  table.insert(inlines, pandoc.Str("."))

  table.insert(doc.blocks, pandoc.HorizontalRule())
  table.insert(doc.blocks, pandoc.Div({ pandoc.Para(inlines) }, pandoc.Attr("", { "crosspost-note" })))
  return doc
end
