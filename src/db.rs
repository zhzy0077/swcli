#![allow(dead_code)]

pub(crate) mod models {
    use serde_json::Value;
    use std::collections::HashMap;

    #[derive(Debug, Clone)]
    pub(crate) struct ProtocolEndpointEntry {
        pub(crate) base_url: String,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct Provider {
        pub(crate) protocol_endpoints: Value,
        pub(crate) default_protocol: String,
    }

    impl Provider {
        pub(crate) fn parsed_protocol_endpoints(&self) -> HashMap<String, ProtocolEndpointEntry> {
            self.protocol_endpoints
                .as_object()
                .map(|object| {
                    object
                        .iter()
                        .filter_map(|(key, value)| {
                            let base_url = value.get("base_url")?.as_str()?.to_string();
                            Some((key.clone(), ProtocolEndpointEntry { base_url }))
                        })
                        .collect()
                })
                .unwrap_or_default()
        }

        pub(crate) fn effective_default_protocol(&self) -> &str {
            &self.default_protocol
        }
    }
}
