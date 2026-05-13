use dialoguer::FuzzySelect;
use dialoguer::theme::ColorfulTheme;
use std::io;

#[derive(Debug, Clone)]
pub struct PickerItem {
    pub label: String,
    pub detail: Option<String>,
}

pub fn pick(prompt: &str, items: &[PickerItem]) -> io::Result<Option<usize>> {
    if items.is_empty() {
        return Ok(None);
    }

    let labels = items
        .iter()
        .map(|item| match item.detail.as_deref() {
            Some(detail) => format!("{}  {}", item.label, detail),
            None => item.label.clone(),
        })
        .collect::<Vec<_>>();

    FuzzySelect::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .items(&labels)
        .default(0)
        .interact_opt()
        .map_err(io::Error::other)
}
