const Load = () => {
	document.querySelector('.btn _load cross-load-more').click()
	setInterval(Load, 1500)
}

Load()
