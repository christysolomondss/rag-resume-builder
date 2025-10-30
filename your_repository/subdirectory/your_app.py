import streamlit as st


def main():
    st.set_page_config(page_title="Your App", layout="centered")
    st.title("Example: your_app.py")
    st.write("This is a small example Streamlit app placed in `your_repository/subdirectory/your_app.py`.")

    name = st.text_input("Your name", value="")
    if st.button("Greet"):
        if name:
            st.success(f"Hello, {name}! 👋")
        else:
            st.info("Enter a name to see a greeting.")


if __name__ == "__main__":
    main()
